import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


# --- 原始全局 mean ---
@libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(inp_val, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


@libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val)


def mean(inp, *, dtype=None):
    logger.debug("GEMS MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


# --- 泛化 2D reduction kernel (用于 fallback 路径) ---
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)


# --- 特化 kernel 1：512x512 case (H100优化) ---
@libentry()
@libtuner(
    configs=[
        # 原始高性能配置
        triton.Config({"BLOCK_M": 4,  "BLOCK_N": 512, "BLOCK_K": 32}, num_warps=4),
        # H100优化配置
        triton.Config({"BLOCK_M": 8,  "BLOCK_N": 512, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 4,  "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 8,  "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def mean_dim_kernel_1(
    inp,
    out_val,
    M,
    N,
    K,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_local = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m = m_local[:, None]
    k = k_local[None, :]

    acc_type = tl.float32
    inv_N = 1.0 / N

    n = tl.arange(0, BLOCK_N)
    offset = m * N * K + n[:, None, None] * K + k[None, :, :]
    inp_vals = tl.load(inp + offset).to(acc_type)

    local_sum = tl.sum(inp_vals, axis=0)
    local_mean = local_sum * inv_N
    local_mean = local_mean.to(dtype)

    out_offset = m * K + k
    tl.store(out_val + out_offset, local_mean)


# --- 特化 kernel 2：1024x1024 case (H100优化) ---
@libentry()
@libtuner(
    configs=[
        # 原始配置
        triton.Config({"BLOCK_M": 8,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=8),
        # H100优化配置
        triton.Config({"BLOCK_M": 8,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 8,  "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 8,  "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 4,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def mean_dim_kernel_2(
    inp,
    out,
    M,
    N,
    K,
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]

    accum_dtype = tl.float32
    sum_val = tl.full((BLOCK_M, BLOCK_K), 0.0, dtype=accum_dtype)

    for pid_n in range(N // BLOCK_N):
        n_start = pid_n * BLOCK_N
        n = n_start + tl.arange(0, BLOCK_N)
        offset = m * N * K + n[:, None, None] * K + k[None, :, :]
        inp_vals = tl.load(inp + offset).to(accum_dtype)
        sum_val += tl.sum(inp_vals, 0)

    mean_val = sum_val / N
    mean_val = mean_val.to(dtype)

    out_offset = m * K + k
    tl.store(out + out_offset, mean_val)


# --- 主函数：mean_dim ---
def mean_dim(x, dim, keepdim=False, *, dtype=None):
    logger.debug("GEMS MEAN DIM")

    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = mean(x, dtype=dtype)
        if not keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]

    N = 1
    K = 1
    M = 1
    if len(x.shape) == 3 and len(dim) == 1 and dim[0] == 1:
        n_dim = dim[0]
        N = list(x.shape)[n_dim]
        M = math.prod(shape[:n_dim])
        K = x.numel() // N // M

    for i in dim:
        shape[i] = 1
    out = torch.empty(shape, dtype=dtype, device=x.device)
    torch2triton_dtype = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}

    # Path 1: (512,512)
    if N == 512 and K == 512 and M % 4 == 0 and len(shape) == 3 and len(dim) == 1 and dim[0] == 1:
        triton_dtype = torch2triton_dtype[x.dtype]
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(K, META["BLOCK_K"]))
        with torch_device_fn.device(x.device):
            mean_dim_kernel_1[grid](
                x, out, M, N, K, dtype=triton_dtype
            )

    # Path 2: (1024,1024)
    elif N == 1024 and K == 1024 and M % 8 == 0 and len(shape) == 3 and len(dim) == 1 and dim[0] == 1:
        triton_dtype = torch2triton_dtype[x.dtype]
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(K, META["BLOCK_K"]))
        with torch_device_fn.device(x.device):
            mean_dim_kernel_2[grid](
                x, out, M, N, K, dtype=triton_dtype
            )

    # Fallback path
    else:
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= x.shape[i]
        M = x.numel() // N
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        with torch_device_fn.device(x.device):
            mean_dim_kernel[grid](x, out, M, N)

    if not keepdim:
        out = out.squeeze(dim)
    return out