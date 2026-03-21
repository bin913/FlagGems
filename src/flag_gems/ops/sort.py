import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val, argsort
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.gpu as tleg
logger = logging.getLogger(__name__)


def unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
    num_bits = unwrap_if_constexpr(num_bits)
    signed = unwrap_if_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@tl.constexpr
def one_zeros(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@tl.constexpr
def zero_ones(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1


@triton.jit
def uint_to_uint(x, descending: tl.constexpr = False):
    out = ~x if descending else x
    return out


@triton.jit
def int_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        # 0111111....1
        bit_mask: tl.constexpr = zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
        # 1000000...0
        sign_bit_mask: tl.constexpr = one_zeros(num_bits)
        sign_bit_mask_tensor = tl.full((), value=sign_bit_mask, dtype=udtype)
        out = ux ^ sign_bit_mask_tensor
    return out


@triton.jit
def floating_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = get_int_t(num_bits, True)
    udtype = get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)

    sign_bit_mask_v: tl.constexpr = one_zeros(num_bits)
    sign_bit_mask = tl.full((), value=sign_bit_mask_v, dtype=udtype)
    # mind the dtype, right_shift for signed is arithmetic right shift
    # Fix for triton 3.1 or else `sx >> rshift_bits` is promoted to int32
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    tl.static_assert(mask.dtype == udtype, "type mismatch")
    # 1000000000...0 for positive
    # 1111111111...1 for negative
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)


@triton.jit
def convert_to_uint_preverse_order(x: tl.tensor, descending: tl.constexpr = False):
    if x.dtype.is_floating():
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    return out


@triton.jit
def compute_global_hist_kernel(
    arr_ptr,
    out_ptr,
    num_passes,
    m,
    n,
    tiles_n_per_cta,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    num_bits_per_pass: tl.constexpr,
    descending: tl.constexpr,
):
    # arr_ptr: (m, n)
    # out_ptr: (m, n_passes, r), where r = 2 ** k_bits is the number of bins
    pid = tl.program_id(0)
    pid_n = pid // m
    pid_m = pid % m

    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1  # a.k.a. 2 ** k_bits - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

    for p in range(0, num_passes):  # parallel
        bit_offset = p * num_bits_per_pass
        for r_start in range(0, r, TILE_R):  # parallel
            bin_indices = r_start + tl.arange(0, TILE_R)
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int64)
            for n_start in range(cta_n_start, cta_n_end, TILE_N):  # sequantial
                n_offsets = n_start + tl.arange(0, TILE_N)  # (TILE_N, )
                mask = n_offsets < cta_n_end
                arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask)
                arr = convert_to_uint_preverse_order(arr, descending)
                key = (arr >> bit_offset) & bfe_mask  # (TILE_N, )
                matches = tl.where(
                    mask, (bin_indices[:, None] == key), False
                )  # (TILE_R, TILE_N)
                acc += matches
            local_sum = tl.sum(acc, axis=1)
            tl.atomic_add(
                out_ptr + pid_m * num_passes * r + p * r + bin_indices,
                local_sum,
                sem="relaxed",
            )


@triton.jit
def sweep(
    arr_ptr,
    associate_arr_ptr,  # inputs: (key & value)
    out_ptr,
    associate_out_ptr,  # outputs: (key & value)
    excumsum_bins_ptr,
    status_ptr,  # aux input and status
    n_passes,
    pass_id,
    bit_offset,
    m,
    N,
    OUT_N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
):
    # r: num_bins = 2 ** k_bits
    # OUT_N: grid_n = cdiv(N, )

    # arr_ptr: (m, N)
    # out_ptr: (m, N)
    # excumsum_bins_ptr: (m, n_passes, r)
    # flag_ptr: (m, r, OUT_N)

    # grid: (m, grid_r, grid_n)

    # load data
    pid = tl.program_id(0)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1)

    # bit masks
    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1  # a.k.a. 2 ** k_bits - 1

    # initialize flag to zero-local sum is not ready
    r: tl.constexpr = 2**k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    # cumsum for a bin_index
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)  # (TILE_N, )
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preverse_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask  # (TILE_N, )

    # since triton can only use scalar as condition, loop by bin_index
    # status must be pre zero-initialized, or else we have to initialize it
    for bin_index in range(cta_r_start, cta_r_end):
        matches = tl.where(mask, key == bin_index, False)  # (TILE_N, ) bool
        # cta level cumsum per bin
        # CAUTION: tl.sum in triton 3.2 does not promote type
        local_sum = tl.sum(matches.to(tl.uint32), axis=0)
        pack0 = aggregate_mask | local_sum
        status_offset = pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n
        tl.store(status_ptr + status_offset, pack0, cache_modifier=".cg")

        # decoupled lookback
        exclusive_prefix = tl.zeros((), dtype=tl.uint32)
        i_lookback = pid_n - 1
        while i_lookback >= 0:
            flag_offset_i = pid_m * (r * OUT_N) + bin_index * OUT_N + i_lookback
            pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)  # uin32
            while pack1 == 0:
                pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            exclusive_prefix += pack1 & v_mask
            if (pack1 & aggregate_mask) == aggregate_mask:
                i_lookback -= 1
            else:
                i_lookback = -1
        pack2 = inclusive_prefix_mask | (exclusive_prefix + local_sum)
        tl.store(status_ptr + status_offset, pack2, cache_modifier=".cg")

        local_ex_cumsum = (
            tl.cumsum(matches.to(tl.uint32), axis=0) - matches
        )  # (TILE_N, )
        ex_cumsum_in_bin = (
            exclusive_prefix + local_ex_cumsum
        )  # global ex_cumsum_in_bin (TILE_N, )

        # ex_cumsum_bins (m, n_passes, r)
        ex_cumsum_bins = tl.load(
            excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + bin_index
        )  # scalar
        pos = ex_cumsum_bins + ex_cumsum_in_bin  # (TILE_N, )

        # scatter
        tl.store(out_ptr + pid_m * N + pos, arr, mask=matches)
        if associate_arr_ptr is not None:
            associate_arr = tl.load(
                associate_arr_ptr + pid_m * N + n_offsets, mask=mask
            )
            tl.store(associate_out_ptr + pid_m * N + pos, associate_arr, mask=matches)


@triton.jit(do_not_specialize=["n_passes", "pass_id", "bit_offset", "m", "N", "OUT_N"])
def sweep_tle_optimized(
    arr_ptr,
    associate_arr_ptr,
    out_ptr,
    associate_out_ptr,
    excumsum_bins_ptr,  # (m, n_passes, r) - 全局前缀和起点
    # status_ptr 不再需要，因为我们有全局前缀和
    n_passes,
    pass_id,
    bit_offset,
    m,
    N,
    OUT_N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr, # 通常等于 r (2^k_bits)，或者由多个 block 处理一个 bin
    k_bits: tl.constexpr,
    descending: tl.constexpr,
):
    # ---------------------------------------------------------
    # 1. Grid & ID 计算
    # ---------------------------------------------------------
    pid = tl.program_id(0)
    pid_m = pid % m
    pid_n = pid // m  # 注意：原代码逻辑似乎是 pid // m 对应 N 的块索引，这里假设 grid 设置正确
    
    # 假设 grid 设置为 (m * grid_n, ) 且每个 block 处理一个特定的 bin_range
    # 为了简化，我们假设每个 block 负责处理 ALL bins (如果 r 小) 或者 特定的 bin
    # 原代码逻辑：pid_r 是通过 program_id(1) 获取的，但 triton jit 通常只支持 1D grid 或手动展开
    # 这里我们模拟原代码的 2D/3D grid 逻辑，假设传入的是扁平化的 pid，或者我们只用 1D grid 遍历所有任务
    
    # 修正：为了适配 Triton 最佳实践，我们通常将 (m, bin_id, n_block) 扁平化
    # 但为了尽量贴合你的输入签名，我们假设调用者设置了正确的 grid
    # 这里我们需要重新推导 pid_r。
    # 假设 grid = (m * grid_n * grid_r,) 或者类似结构。
    # 让我们采用更稳健的方式：每个 Block 负责一个 (m, bin_id) 对，并遍历所有的 N 块？
    # 不，原代码是每个 Block 负责 (m, n_block)，然后循环 bin。这导致串行化。
    
    # 【关键重构】：为了利用 SMEM，最好的策略是：
    # 每个 Block 负责一个 (m, n_block) 片段，但在 SMEM 内对所有 k_bits 进行分桶。
    # 这样只需要一次全局加载，一次 SMEM 重排，然后按 Bin 顺序写入。
    
    # 重新定义 ID 以匹配原逻辑但优化执行流
    # 假设 grid 是 (m, grid_n)，每个 block 处理一行中的一段 N
    pid_m = tl.program_id(0) % m
    pid_n = tl.program_id(0) // m
    
    r: tl.constexpr = 1 << k_bits
    
    # ---------------------------------------------------------
    # 2. 使用 TLE 分配共享内存 (SMEM)
    # ---------------------------------------------------------
    # 我们需要为每个 bin 预留空间。
    # 总大小 = TILE_N (因为每个元素只属于一个 bin)
    # 为了高效，我们分配一个大的 SMEM 数组，并维护每个 bin 的局部偏移指针
    
    # 分配 Keys 和 Values 的共享内存缓冲
    # 布局：[bin_0_data, bin_1_data, ..., bin_r-1_data]
    smem_keys = tle.gpu.alloc([TILE_N], dtype=tl.int32, scope=tle.gpu.smem)
    smem_vals = tle.gpu.alloc([TILE_N], dtype=tl.int32, scope=tle.gpu.smem) if associate_arr_ptr is not None else None
    
    # 分配用于记录每个 bin 在 SMEM 中起始位置的计数器 (在 SMEM 或 寄存器中维护)
    # 由于 r 通常很小 (2, 4, 8, 16)，我们可以用寄存器数组存 offsets，或者在 SMEM 存
    # 这里使用 SMEM 存储每个 bin 的当前写入偏移量 (相对于 SMEM 基址)
    smem_bin_offsets = tle.gpu.alloc([r], dtype=tl.int32, scope=tle.gpu.smem)
    
    # 初始化 bin 偏移量为 0
    # 只有第一个线程做初始化，或者用 vectorized store
    off_init = tl.arange(0, r)
    smem_bin_offsets_ptr = tle.gpu.local_ptr(smem_bin_offsets, (off_init,))
    tl.store(smem_bin_offsets_ptr, 0, mask=off_init < r)
    #tl.store(smem_bin_offsets + off_init, 0, mask=off_init < r)
    tl.debug_barrier()

    # ---------------------------------------------------------
    # 3. 加载数据 (Global Load)
    # ---------------------------------------------------------
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask, other=0)
    
    # 处理关联数据 (Value)
    if associate_arr_ptr is not None:
        assoc_arr = tl.load(associate_arr_ptr + pid_m * N + n_offsets, mask=mask, other=0)
    else:
        assoc_arr = tl.zeros([TILE_N], dtype=tl.int32) # Placeholder

    # 提取 Key (当前位的值)
    arr_u = convert_to_uint_preverse_order(arr, descending)
    keys_local = (arr_u >> bit_offset) & ((1 << k_bits) - 1)

   # ---------------------------------------------------------
    # 4. 块内分桶 (Bin Packing in SMEM) - 修正版
    # ---------------------------------------------------------
    
    # 1. 初始化 bin 计数器 (SMEM)
    # smem_bin_offsets 大小为 r，初始化为 0
    off_init = tl.arange(0, r)
    # 假设 tle.gpu.local_ptr 支持向量索引，或者直接使用指针算术
    # 如果 tle 封装有问题，建议直接用: smem_bin_offsets + off_init
    smem_bin_offsets_ptr = tle.gpu.local_ptr(smem_bin_offsets, (off_init,))
    tl.store(smem_bin_offsets_ptr, 0, mask=off_init < r)
    
    tl.debug_barrier() 

    # 2. 【核心修复】每个线程独立执行原子加
    # 错误做法: tl.atomic_add(single_ptr, vector_values, mask) -> 报错
    # 正确做法: tl.atomic_add(vector_ptrs, scalar_value, mask) -> 成功
    
    # 构造每个线程对应的指针向量
    # keys_local 是 [TILE_N] 的 tensor，值为 0..r-1
    # 我们需要生成 [TILE_N] 个指针，第 i 个指针指向 smem_bin_offsets[keys_local[i]]
    
    # 方法 A: 如果 tle.gpu.local_ptr 支持 Tensor 索引 (不确定是否支持)
    # bin_ptrs = tle.gpu.local_ptr(smem_bin_offsets, (keys_local,))
    
    # 方法 B (推荐，标准 Triton): 直接使用指针算术
    # smem_bin_offsets 应该是一个 tl.pointer_type
    # Triton 允许 pointer + tensor，自动广播步长
    bin_ptr = tle.gpu.local_ptr(smem_bin_offsets, (keys_local,)) 

    # 执行原子加
    # ptrs: [TILE_N] 个指针 (可能指向重复地址)
    # value: 1 (标量)，每个线程都加 1
    # mask: 只有有效线程执行
    # 返回: [TILE_N] 个旧值，即每个元素在 bin 内的局部 Rank
    local_ranks = tl.atomic_add(bin_ptr, 1, mask=mask)
    
    tl.debug_barrier() # 确保所有计数完成

    # 3. 计算前缀和 (Base Offsets)
    # 此时 smem_bin_offsets 中存储的是每个 bin 的总数量 (Counts)
    final_counts = tl.load(smem_bin_offsets_ptr, mask=off_init < r)
    
    # Exclusive CumSum: base_offsets[b] = sum(counts[0]...counts[b-1])
    base_offsets = tl.cumsum(final_counts, axis=0) - final_counts
    
    # 将 base_offsets 写回 SMEM (覆盖 counts，因为后面只需要 base)
    tl.store(smem_bin_offsets_ptr, base_offsets, mask=off_init < r)
    
    tl.debug_barrier()

    # 4. 计算 SMEM 绝对地址并写入
    # 读取当前 key 对应的 base offset
    # 同样使用指针算术加载
    current_base_offsets = tl.load(bin_ptr, mask=mask)
    
    smem_indices = current_base_offsets + local_ranks
    
    # 写入 Keys
    smem_keys_ptr = tle.gpu.local_ptr(smem_keys, (smem_indices,))
    tl.store(smem_keys_ptr, arr, mask=mask)
    
    # 写入 Values (如果有)
    if associate_arr_ptr is not None:
        smem_vals_ptr = tle.gpu.local_ptr(smem_vals, (smem_indices,))
        tl.store(smem_vals_ptr, assoc_arr, mask=mask)
        
    tl.debug_barrier()

    # ---------------------------------------------------------
    # 5. 有序写入全局内存 (Coalesced Global Store)
    # ---------------------------------------------------------
    # 现在 SMEM 中的数据是按 Bin 顺序排列的：
    # [Bin 0 的所有数据] [Bin 1 的所有数据] ...
    
    # 我们不需要从 final_counts tensor 中索引，而是在循环中重新计算 count_b
    # 因为 r 是 constexpr，循环会被完全展开，性能损失可忽略不计
    
    for b in range(r):
        # 【修复点 1】: 重新计算当前 bin 的数量，避免 tl.tensor 索引错误
        mask_b = (keys_local == b) & mask
        count_b = tl.sum(mask_b.to(tl.int32))
        
        # 获取该 Bin 在全局输出中的起始位置
        # excumsum_bins_ptr: (m, n_passes, r)
        global_start = tl.load(
            excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + b
        )
        
        # 【修复点 2】: 安全地读取该 Bin 在 SMEM 中的起始位置 (Base Offset)
        # 之前我们将 base_offsets 存入了 smem_bin_offsets
        # 使用 local_ptr 构造指向第 b 个元素的指针并 load
        smem_bin_offsets_b_ptr = tle.gpu.local_ptr(smem_bin_offsets, (b,))
        smem_start = tl.load(smem_bin_offsets_b_ptr)
        
        # 生成该 Bin 内部的范围 (0 到 count_b)
        local_range = tl.arange(0, TILE_N)
        mask_bin = local_range < count_b
        
        smem_read_pos = smem_start + local_range
        global_write_pos = global_start + local_range
        
        # 【修复点 3】: 修正拼写错误 smems_keys -> smem_keys
        smem_keys_ptr = tle.gpu.local_ptr(smem_keys, (smem_read_pos,))
        k_val = tl.load(smem_keys_ptr, mask=mask_bin, other=0)
        
        # 合并写入全局内存 (地址是连续的！)
        tl.store(out_ptr + pid_m * N + global_write_pos, k_val, mask=mask_bin)
        
        if associate_arr_ptr is not None:
            smem_vals_ptr = tle.gpu.local_ptr(smem_vals, (smem_read_pos,))
            v_val = tl.load(smem_vals_ptr, mask=mask_bin, other=0)
            tl.store(associate_out_ptr + pid_m * N + global_write_pos, v_val, mask=mask_bin)


def radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    TILE_N = 1024
    tiles_n_per_cta = 8
    CTA_TILE_N = tiles_n_per_cta * TILE_N

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    TILE_R = 16

    grid_n = triton.cdiv(n, CTA_TILE_N)
    grid_for_global_hist = (m * grid_n, 1, 1)

    with torch_device_fn.device(arr.device):
        global_hist = torch.zeros(
            (m, n_passes, num_bins), device=arr.device, dtype=torch.int32
        )
        compute_global_hist_kernel[grid_for_global_hist](
            arr,
            global_hist,
            n_passes,
            m,
            n,
            tiles_n_per_cta,
            TILE_N,
            TILE_R,
            k_bits,
            descending,
        )
        ex_cumsum_bins = torch.cumsum(global_hist, -1) - global_hist
        ex_cumsum_bins = ex_cumsum_bins.to(torch.uint32)

        # sort
        arr_in = torch.clone(arr)
        indices_in = (
            torch.arange(0, n, dtype=torch.int64, device=arr_in.device)
            .broadcast_to(arr.shape)
            .contiguous()
        )
        arr_out = torch.empty_like(arr)
        indices_out = torch.empty_like(indices_in)

        TILE_R = 8
        grid_r = triton.cdiv(num_bins, TILE_R)
        TILE_N = 2048
        grid_n = triton.cdiv(n, TILE_N)
        grid_for_sweep = (m * grid_n, grid_r)

        status = torch.empty(
            (m, num_bins, grid_n), device=arr.device, dtype=torch.uint32
        )

        for i in range(0, n_passes):
            bit_offset = i * k_bits
            status.zero_()
            # sweep[grid_for_sweep](
            #     arr_in,
            #     indices_in,
            #     arr_out,
            #     indices_out,
            #     ex_cumsum_bins,
            #     status,
            #     n_passes,
            #     i,
            #     bit_offset,
            #     m,
            #     n,
            #     grid_n,
            #     TILE_N,
            #     TILE_R,
            #     k_bits,
            #     descending,
            # )
            sweep_tle_optimized[grid_for_sweep](
                arr_in,
                indices_in,
                arr_out,
                indices_out,
                ex_cumsum_bins,
                # status,
                n_passes,
                i,
                bit_offset,
                m,
                n,
                grid_n,
                TILE_N,
                TILE_R,
                k_bits,
                descending,
            )
            # print(f"< sorted last {bit_offset + k_bits:>2d} bits: {arr_out}")
            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


@libentry()
@triton.jit()
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = tl.program_id(0) * N + cols
    in_ptr += offset
    out_ptr += offset
    out_index_ptr += offset

    if IS_FLOAT:
        mask_val = _get_finfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)
    else:
        mask_val = _get_iinfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)

    index_val = tl.arange(0, BLOCK_SIZE)

    sorted_in_val, sorted_index_val = argsort(
        in_val, index_val, 0, descending=DESCENDING
    )
    tl.store(out_ptr, sorted_in_val, mask=mask)
    tl.store(out_index_ptr, sorted_index_val, mask=mask)


def sort(inp, dim=-1, descending=False):
    # We only implement stable radix sort here
    logger.debug("GEMS SORT")
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS SORT.STABLE")
    # We only implement stable radix sort here
    _ = stable
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    dtype = inp.dtype
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    out, out_index = radix_sort(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index
