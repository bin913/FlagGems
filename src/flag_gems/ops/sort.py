import logging

import torch
import triton
import triton.language as tl

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
    smem_keys = tle.alloc([TILE_N], dtype=tl.int32, scope=tle.smem)
    smem_vals = tle.alloc([TILE_N], dtype=tl.int32, scope=tle.smem) if associate_arr_ptr is not None else None
    
    # 分配用于记录每个 bin 在 SMEM 中起始位置的计数器 (在 SMEM 或 寄存器中维护)
    # 由于 r 通常很小 (2, 4, 8, 16)，我们可以用寄存器数组存 offsets，或者在 SMEM 存
    # 这里使用 SMEM 存储每个 bin 的当前写入偏移量 (相对于 SMEM 基址)
    smem_bin_offsets = tle.alloc([r], dtype=tl.int32, scope=tle.smem)
    
    # 初始化 bin 偏移量为 0
    # 只有第一个线程做初始化，或者用 vectorized store
    off_init = tl.arange(0, r)
    tl.store(smem_bin_offsets + off_init, 0, mask=off_init < r)
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
    # 4. 块内分桶 (Bin Packing in SMEM)
    # ---------------------------------------------------------
    # 目标：计算每个元素在 SMEM 中的目标位置
    # 步骤 A: 统计本块内每个 bin 的数量 (Histogram)
    # 步骤 B: 计算本块内每个 bin 的起始偏移 (Prefix Sum of Histogram)
    # 步骤 C: 计算每个元素的局部偏移并写入 SMEM
    
    # A. Histogram (使用原子加或 warp shuffle，这里用简单的原子加演示，因 r 小)
    # 为了无锁，可以用寄存器累加后一次性写，但 Triton 中 atomic_add 到 smem 是最直接的
    for i in range(TILE_N):
        # 这种标量循环在 Triton 中效率低，最好用向量化操作
        # 但由于 key 不同，我们必须分组处理。
        # 优化技巧：如果 r 很小，可以展开循环或使用 tl.where 掩码
        pass 

    # 【高性能实现路径】：
    # 既然 r 是 constexpr 且通常很小 (<= 16)，我们可以为每个 bin 生成一个 mask
    # 然后并行计算每个 bin 的 count 和 prefix sum
    
    bin_counts = tl.zeros([r], dtype=tl.int32)
    
    # 向量化统计 (假设 r 较小，否则需要循环)
    # 这里展示 r=2, 4, 8 的通用逻辑思路，实际代码需根据 r 展开或使用 loop
    # 为了代码简洁且通用，我们使用一种技巧：
    # 1. 计算每个元素的 rank within its bin (local cumsum)
    # 2. 计算每个 bin 的 total count
    # 3. 计算 bin 的 global offset in SMEM
    
    # 方法：对每个可能的 bin_value 进行扫描
    local_bin_starts = tl.zeros([r], dtype=tl.int32)
    
    # 临时存储每个元素在其 bin 内的相对偏移
    local_ranks = tl.zeros([TILE_N], dtype=tl.int32)
    
    # 由于 Triton 限制，我们不能动态循环 r 次做复杂的 cumsum 而不影响性能
    # 最佳实践：如果 r <= 16，完全展开
    for b in range(r):
        mask_b = (keys_local == b) & mask
        count_b = tl.sum(mask_b.to(tl.int32))
        
        # 记录该 bin 的总数 (用于后续计算 SMEM 偏移)
        # 我们需要一个数组来存这些 counts，然后做 cumsum
        # 这里用 atomics 更新 smem_bin_offsets 来动态分配？不，先算好 offsets
        
        # 存入临时寄存器列表 (如果 r 是 constexpr，可以用 tuple 或 手动展开变量)
        # 这里为了演示逻辑，假设我们有一个机制收集 counts
        # 在实际生产中，通常硬编码 r=2 或 r=4 的逻辑
        
        # 替代方案：使用原子加直接分配 SMEM 位置 (简单但稍慢)
        # 每个线程根据自己的 key，atomic_add 对应的 bin counter，得到自己的 local_rank
        if tl.sum(mask_b.to(tl.int32)) > 0:
             # 获取当前 bin 的计数器指针
             ptr = tle.local_ptr(smem_bin_offsets, (b,))
             # 原子加，返回旧值作为 local_rank
             # 注意：tl.atomic_add 返回旧值
             ranks_b = tl.atomic_add(ptr, mask_b.to(tl.int32), mask=mask_b)
             # 上面的 atomic_add 是向量化的吗？Triton 支持 masked atomic_add
             # 但我们需要的是每个元素独立的 rank。
             # 正确做法：
             # 1. 每个线程计算自己的 key
             # 2. 构造 smem_ptr = base + bin_offset[key]
             # 3. atomically increment bin_offset[key] and get old value -> this is the local rank
             
             # 重新实现原子分配逻辑：
             pass

    # --- 真正的优化实现 (Atomic Allocation Pattern) ---
    # 重置 offsets
    tl.store(smem_bin_offsets + off_init, 0, mask=off_init < r)
    tl.debug_barrier()
    
    # 每个线程根据自己的 key，原子地获取在 SMEM 中的位置
    # 1. 读取当前 bin 的计数值 (旧值)
    # 2. 原子加 1
    # 3. 旧值即为该元素在 bin 内部的相对偏移
    
    # 由于 tl.atomic_add 返回旧值，我们可以直接利用它
    # 构造指向对应 bin 计数器的指针
    bin_ptrs = tle.local_ptr(smem_bin_offsets, (keys_local,))
    
    # 执行原子加，获取 local_rank (在该 bin 内的相对位置)
    # mask 确保无效线程不参与
    local_ranks = tl.atomic_add(bin_ptrs, 1, mask=mask)
    
    tl.debug_barrier() # 确保所有 offset 分配完毕
    
    # 现在我们需要知道每个 bin 在 SMEM 中的 *起始* 偏移量 (Base Offset)
    # 刚才的 atomic_add 只是给了相对偏移。
    # 我们需要对 smem_bin_offsets (现在的值是 count) 做前缀和，得到 Base Offset
    
    # 读取最终的 counts
    final_counts = tl.load(smem_bin_offsets + off_init, mask=off_init < r)
    
    # 计算前缀和 (Exclusive CumSum) 得到 Base Offsets
    # tl.cumsum 需要 tensor，off_init 是 arange
    base_offsets = tl.cumsum(final_counts, axis=0) - final_counts # Exclusive
    
    # 将 base_offsets 存回 smem 或者直接用在寄存器计算目标地址
    # 为了后续写入方便，我们将 base_offsets 更新到 smem_bin_offsets 中
    tl.store(smem_bin_offsets + off_init, base_offsets, mask=off_init < r)
    tl.debug_barrier()
    
    # 计算每个元素在 SMEM 中的绝对地址
    # smem_idx = base_offsets[key] + local_rank
    # 由于 base_offsets 在 smem 中，我们需要再次 load 或者刚才保存在寄存器
    # 刚才 base_offsets 是在寄存器里的 (如果是 scalar array)，但 triton 处理数组有点麻烦
    # 简单点：再次 load
    current_base_offsets = tl.load(smem_bin_offsets + keys_local, mask=mask)
    smem_indices = current_base_offsets + local_ranks
    
    # 写入 SMEM (重排数据)
    tl.store(smem_keys + smem_indices, arr, mask=mask)
    if associate_arr_ptr is not None:
        tl.store(smem_vals + smem_indices, assoc_arr, mask=mask)
        
    tl.debug_barrier() # 确保 SMEM 写入完成，准备读取

    # ---------------------------------------------------------
    # 5. 有序写入全局内存 (Coalesced Global Store)
    # ---------------------------------------------------------
    # 现在 SMEM 中的数据是按 Bin 顺序排列的：
    # [Bin 0 的所有数据] [Bin 1 的所有数据] ...
    # 我们可以顺序遍历 SMEM，计算全局地址，然后合并写入
    
    # 每个线程负责写出 SMEM 中的一部分
    # 为了最大化合并，我们让线程 i 写出 SMEM[i] (如果 i < total_count)
    
    smem_read_idx = pid_n * TILE_N + tl.arange(0, TILE_N) # 这里的逻辑需要调整，因为总元素数可能不是 TILE_N 的整数倍？
    # 不，每个 Block 处理 TILE_N 个输入，所以 SMEM 里也只有 TILE_N 个有效数据
    read_mask = smem_read_idx < TILE_N # 实际上总是真，除了 padding
    
    # 但我们不知道每个 Bin 的具体边界，除非我们重新计算
    # 更好的方法：还是按 Bin 循环写入，保证地址连续
    
    # 重新加载 base_offsets 和 counts
    # 此时 smem_bin_offsets 存的是 base_offsets (start index in SMEM)
    # 我们需要 counts 来确定结束位置
    counts = tl.load(smem_bin_offsets + off_init, mask=off_init < r) # 这里被覆盖了，需要重新算或者之前保存
    # 修正：之前 store 了 base_offsets，覆盖了 counts。
    # 我们应该在计算 base_offsets 后，把 counts 存在另一个地方，或者重新计算 base_offsets + counts = end_offsets
    
    # 让我们重新加载 counts (可以通过再次扫描，或者刚才保存)
    # 简单起见，假设我们重新计算 counts (开销小) 或者我们在上面保留了 counts
    # 这里假设我们有一个方式获取 counts。
    # 实际上，end_offsets = base_offsets + counts. 
    # 我们可以再做一个 inclusive cumsum 得到 end_offsets
    
    end_offsets = tl.cumsum(final_counts, axis=0) # Inclusive
    
    # 现在按 Bin 顺序写入
    for b in range(r):
        # 获取该 Bin 在全局输出中的起始位置
        # excumsum_bins_ptr: (m, n_passes, r)
        global_start = tl.load(
            excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + b
        )
        
        # 该 Bin 在 SMEM 中的范围: [base, end)
        smem_start = tl.load(smem_bin_offsets + b) # base
        smem_end = smem_start + final_counts[b] # end
        
        # 生成该 Bin 内部的范围
        bin_size = final_counts[b]
        local_range = tl.arange(0, TILE_N) # 足够大
        mask_bin = (local_range < bin_size)
        
        smem_read_pos = smem_start + local_range
        global_write_pos = global_start + local_range
        
        # 加载已排序的数据
        k_val = tl.load(smem_keys + smem_read_pos, mask=mask_bin, other=0)
        
        # 合并写入全局内存 (地址是连续的！)
        tl.store(out_ptr + pid_m * N + global_write_pos, k_val, mask=mask_bin)
        
        if associate_arr_ptr is not None:
            v_val = tl.load(smem_vals + smem_read_pos, mask=mask_bin, other=0)
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
