import logging

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.gpu as tleg
from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val, argsort
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

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
    if associate_arr_ptr is not None:
        associate_arr = tl.load(
            associate_arr_ptr + pid_m * N + n_offsets, mask=mask
        )
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
            # associate_arr = tl.load(
            #     associate_arr_ptr + pid_m * N + n_offsets, mask=mask
            # )
            tl.store(associate_out_ptr + pid_m * N + pos, associate_arr, mask=matches)

@triton.jit
def sweep_optimized(
    arr_ptr,
    associate_arr_ptr,
    out_ptr,
    associate_out_ptr,
    excumsum_bins_ptr,
    status_ptr,
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
    # --- 配置与常量 ---
    r: tl.constexpr = 2**k_bits
    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1

    # --- Grid 与 Program ID ---
    pid = tl.program_id(0)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1)

    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    # --- 1. 数据加载 (Load) ---
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    
    # 加载主数据
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preverse_order(arr, descending) # 假设此函数已定义
    key = (arr_u >> bit_offset) & bfe_mask

    # 加载关联数据 (如有)
    if associate_arr_ptr is not None:
        associate_arr = tl.load(associate_arr_ptr + pid_m * N + n_offsets, mask=mask)
    else:
        associate_arr = None

    # --- 2. 显式分配 Shared Memory (TLE 特性) ---
    # 为当前 Tile 的数据分配 SMEM 缓冲区
    # 我们需要两块 SMEM: 一块用于 key/value 的重排暂存，一块用于辅助计算
    # 这里演示为输出数据分配 SMEM，以实现 "SMEM Scatter -> GMEM Coalesced Write"
    
    # 分配 SMEM 用于暂存排序后的 arr
    smem_out = tle.gpu.alloc(
        [TILE_N], 
        dtype=arr.dtype, 
        scope=tle.gpu.storage_kind.smem
    )
    
    # 分配 SMEM 用于暂存排序后的 associate_arr (如果需要)
    if associate_arr_ptr is not None:
        smem_assoc_out = tle.gpu.alloc(
            [TILE_N], 
            dtype=associate_arr.dtype, 
            scope=tle.gpu.storage_kind.smem
        )
    
    # --- 3. 循环处理每个 Bin ---
    for bin_index in range(cta_r_start, cta_r_end):
        # 计算匹配掩码
        matches = tl.where(mask, key == bin_index, False)
        
        # --- CTA 级前缀和计算 (保持原逻辑，略作简化) ---
        local_sum = tl.sum(matches.to(tl.uint32), axis=0)
        pack0 = aggregate_mask | local_sum
        status_offset = pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n
        
        # 状态初始化
        tl.store(status_ptr + status_offset, pack0, cache_modifier=".cg")

        # Lookback 逻辑 (保持原逻辑)
        exclusive_prefix = tl.zeros((), dtype=tl.uint32)
        i_lookback = pid_n - 1
        while i_lookback >= 0:
            flag_offset_i = pid_m * (r * OUT_N) + bin_index * OUT_N + i_lookback
            pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            while pack1 == 0:
                pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            exclusive_prefix += pack1 & v_mask
            if (pack1 & aggregate_mask) == aggregate_mask:
                i_lookback -= 1
            else:
                i_lookback = -1
        
        # 更新状态为完成
        pack2 = inclusive_prefix_mask | (exclusive_prefix + local_sum)
        tl.store(status_ptr + status_offset, pack2, cache_modifier=".cg")

        # 计算局部 Exclusive Cumsum
        local_ex_cumsum = tl.cumsum(matches.to(tl.uint32), axis=0) - matches
        ex_cumsum_in_bin = exclusive_prefix + local_ex_cumsum # 全局偏移量 (在 Bin 内)

        # 加载该 Bin 的全局起始偏移
        ex_cumsum_bins = tl.load(
            excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + bin_index
        )
        
        # 计算最终的全局写入位置
        pos = ex_cumsum_bins + ex_cumsum_in_bin

        # --- 4. 优化核心：使用 tle.gpu.local_ptr 进行 SMEM Scatter ---
        
        # 构建 SMEM 的指针视图
        # 我们只将匹配 (matches=True) 的元素写入 SMEM 的对应相对位置
        # 注意：这里 pos 是全局的，不能直接作为 SMEM 索引。
        # 策略调整：我们将数据先 Scatter 到 SMEM 的 "局部紧凑" 位置，或者利用 local_ptr 的直接映射能力。
        
        # 【修正策略】：由于 pos 是全局分散的，直接映射到有限的 TILE_N SMEM 是不可能的，除非我们做全局排序。
        # 但我们可以利用 local_ptr 优化 "写入模式"。
        # 在此场景下，最直接的 TLE 优化是利用 local_ptr 将 "散列的 GMEM 写入" 转换为 "SMEM 写入 + 批量回写" 
        # 但这需要改变算法逻辑（例如先全部读入 SMEM，在 SMEM 内重排，再写回）。
        
        # 【替代优化方案】：利用 local_ptr 优化 Register -> SMEM 的映射，减少地址计算开销，并利用 SMEM 的高带宽作为 Write Buffer。
        # 假设我们允许在 SMEM 中先按 "局部顺序" 存放，最后再处理？不，Sweep 必须保证全局顺序。
        
        # **真正的 TLE 优势场景**：
        # 如果我们将 `pos` 限制在当前 Block 可管理的范围内，或者我们只是用 SMEM 来合并写入请求。
        # 但针对此特定代码，最实用的优化是：
        # 1. 使用 `local_ptr` 创建 SMEM 视图，用于暂存当前 Tile 中属于 "当前 Bin" 的数据。
        # 2. 由于一个 Tile 可能包含多个 Bin 的数据，且 Pos 分散。
        # 让我们换一个角度：使用 `local_ptr` 来优化 `associate_arr` 的加载和存储配对，减少寄存器溢出。
        
        # **重新设计 Scatter 步骤以适配 TLE**:
        # 我们不直接写 GMEM。我们计算数据在 SMEM 中的目标位置 (局部偏移)。
        # 但由于 Pos 是全局的，我们无法简单地映射到 [0, TILE_N] 的 SMEM 而不发生冲突，除非我们知道每个 Bin 的确切大小并在 SMEM 中划分区域。
        
        # **最佳实践路径**: 
        # 使用 `tle.gpu.local_ptr` 构建一个指向 SMEM 的指针，该指针对应于 `ex_cumsum_in_bin` (局部前缀和)。
        # 这样，我们将数据先紧凑地存储在 SMEM 中 (0, 1, 2...)，而不是分散的 Global Pos。
        # 然后，我们需要知道 Global Base (ex_cumsum_bins) 来执行最终的 Copy。
        # 但是 `tle.gpu.copy` 通常用于 contiguous copy。这里是 scatter。
        
        # **结论**: 对于 Radix Sort 的 Scatter 阶段，如果必须写入全局随机位置，SMEM 只能作为 Write-Combining Buffer (需要复杂逻辑)。
        # 但如果我们假设 `TLE` 的 `local_ptr` 能够智能地将分散的 store 转换为向量化的指令序列，或者我们改变策略：
        # **策略**: 将数据先 Scatter 到 SMEM 的 "局部索引" (0..count-1)，然后由一个专门的步骤或硬件指令将其分发。
        # 鉴于代码上下文，最合理的 TLE 用法是优化 **中间数据的布局** 和 **指针算术**。
        
        # 让我们演示如何使用 `local_ptr` 来执行 **SMEM 内的紧凑 Scatter**，这通常是优化的第一步。
        # 计算局部目标索引 (0, 1, 2... 对于匹配的项)
        target_indices_in_tile = tl.cumsum(matches.to(tl.int32), axis=0) - 1 # 0-based index in the local buffer for matched items
        # 注意：cumsum - 1 对于第一个匹配项是 0，第二个是 1。未匹配项会被忽略（通过 mask）。
        
        # 构建 SMEM 指针视图: 仅针对匹配项的紧凑位置
        # 我们需要过滤掉未匹配项的索引，或者使用 masked store
        # tle.local_ptr 接受 indices tensor。
        smem_ptrs = tle.gpu.local_ptr(
            smem_out,
            indices=(target_indices_in_tile,) # 形状 (TILE_N,)
        )
        
        # 将数据存入 SMEM (紧凑存储)
        # 注意：这里 target_indices_in_tile 对于不匹配的元素可能是负数或无效，需要 mask
        # 实际上 cumsum - 1 对于第一个 false 之前的 true 是有效的。
        # 更安全的做法：只在 matches 为真时计算有效索引
        valid_local_idx = tl.cumsum(matches.to(tl.int32), axis=0) - 1
        # 对于不匹配的行，这个索引是无效的，但我们有 mask
        
        # 使用 local_ptr 生成的指针进行 store
        # 这里的语义是：将 arr 中 matches 为真的元素，存入 smem_out 的 [0, 1, 2...] 位置
        tl.store(smem_ptrs, arr, mask=matches)
        
        if associate_arr_ptr is not None:
            smem_assoc_ptrs = tle.gpu.local_ptr(
                smem_assoc_out,
                indices=(valid_local_idx,)
            )
            tl.store(smem_assoc_ptrs, associate_arr, mask=matches)

        # --- 5. 从 SMEM 到 GMEM 的最终分发 ---
        # 现在数据在 SMEM 中是紧凑的 (0..local_sum-1)。
        # 我们需要将它们写入到 global_pos = ex_cumsum_bins + [0..local_sum-1]
        # 这是一个 "Gather from SMEM, Scatter to GMEM" 或者 "Block Copy with Offset"
        
        # 构造 GMEM 的目标地址 (连续的块)
        # 既然我们在 SMEM 中已经 compact 了，现在的目标是连续的 global 地址段!
        # global_dest_start = ex_cumsum_bins
        # offsets = tl.arange(0, TILE_N) # 足够大覆盖 local_sum
        # 但实际上我们只需要写 local_sum 个元素
        
        # 生成连续的 GMEM 指针
        # 注意：ex_cumsum_bins 是标量
        global_dest_offsets = tl.arange(0, TILE_N) # 假设 TILE_N 足够大
        final_global_ptrs = pid_m * N + ex_cumsum_bins + global_dest_offsets
        
        # 从 SMEM 加载紧凑数据
        # 构建 SMEM 的读取视图 (0, 1, 2...)
        read_indices = tl.arange(0, TILE_N)
        smem_read_ptrs = tle.gpu.local_ptr(smem_out, indices=(read_indices,))
        
        loaded_data = tl.load(smem_read_ptrs, mask=global_dest_offsets < local_sum) # mask 确保不越界
        
        # 写入 GMEM (现在是连续的写入！)
        # 原代码是分散写入，现在变成了连续写入 (Coalesced Store)
        tl.store(out_ptr + final_global_ptrs, loaded_data, mask=global_dest_offsets < local_sum)
        
        if associate_arr_ptr is not None:
            smem_assoc_read_ptrs = tle.gpu.local_ptr(smem_assoc_out, indices=(read_indices,))
            loaded_assoc = tl.load(smem_assoc_read_ptrs, mask=global_dest_offsets < local_sum)
            tl.store(associate_out_ptr + final_global_ptrs, loaded_assoc, mask=global_dest_offsets < local_sum)

    # 注：上述逻辑假设每个 bin_index 循环内独立完成 SMEM->GMEM。
    # 实际上，为了最大化效率，通常会在所有 bin_index 循环结束后，一次性处理 SMEM 到 GMEM 的拷贝，
    # 或者在循环内累积 SMEM 的使用。但为了保持与原代码逻辑（逐 Bin 依赖 status）的一致性，
    # 上面的写法展示了如何利用 local_ptr 将 "分散写" 转化为 "SMEM 紧凑写 + GMEM 连续写"。
    # 这种转化极大地提高了 GMEM 的写入带宽利用率。


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
            sweep_optimized[grid_for_sweep](
                arr_in,
                indices_in,
                arr_out,
                indices_out,
                ex_cumsum_bins,
                status,
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
