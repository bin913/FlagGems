import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


def is_tma_available():
    """Check if TMA (Tensor Memory Accelerator) is available."""
    try:
        # Check if triton.tools.tensor_descriptor module exists
        from triton.tools.tensor_descriptor import TensorDescriptor
        return True
    except ImportError:
        return False


def is_tma_compatible(bias, mat1, mat2, N, K):
    """
    Check if tensors are compatible with TMA (Tensor Memory Accelerator).

    TMA requires 128-bit (16-byte) alignment for memory access:
    - For FP16/BF16 (2 bytes/element): N and K must be multiples of 8
      (8 elements × 2 bytes = 16 bytes)
    - For FP32 (4 bytes/element): N and K must be multiples of 4
      (4 elements × 4 bytes = 16 bytes)

    Args:
        bias, mat1, mat2: Input tensors
        N, K: Matrix dimensions

    Returns:
        bool: True if compatible with TMA's 128-bit alignment requirement
    """
    if not is_tma_available():
        return False

    return (
        mat1.dtype in (torch.float16, torch.bfloat16)
        and mat2.dtype in (torch.float16, torch.bfloat16)
        and bias.dtype in (torch.float16, torch.bfloat16)
        and N % 8 == 0
        and K % 8 == 0
    ) or (
        mat1.dtype in (torch.float32,)
        and mat2.dtype in (torch.float32,)
        and bias.dtype in (torch.float32,)
        and N % 4 == 0
        and K % 4 == 0
    )


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("addmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel_general(
    a_ptr,
    b_ptr,
    i_ptr,
    c_ptr,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        if a.dtype == tl.float16 or a.dtype == tl.bfloat16:
            accumulator += tl.dot(a, b, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, input_precision="tf32x3")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    i_ptrs = i_ptr + stride_im * offs_cm[:, None] + stride_in * offs_cn[None, :]
    bias = tl.load(i_ptrs, mask=c_mask, other=0.0)

    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


def addmm_tma_set_block_size_hook(nargs):
    BLOCK_SIZE_M = nargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = nargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = nargs["BLOCK_SIZE_K"]
    if nargs["A_ROW_MAJOR"]:
        nargs["a_desc"].block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_K]
    else:
        nargs["a_desc"].block_shape = [BLOCK_SIZE_K, BLOCK_SIZE_M]

    if nargs["B_ROW_MAJOR"]:
        nargs["b_desc"].block_shape = [BLOCK_SIZE_K, BLOCK_SIZE_N]
    else:
        nargs["b_desc"].block_shape = [BLOCK_SIZE_N, BLOCK_SIZE_K]

    nargs["c_desc"].block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N]
    nargs["i_desc"].block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N]


def addmm_get_configs(pre_hook=addmm_tma_set_block_size_hook):
    return [
        triton.Config(
            {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK},
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in [32, 64, 128, 256]
        for BN in [32, 64, 128]
        for BK in [32, 64, 128]
        for s in [2, 3, 4]
        for w in [4, 8]
    ]


@libentry()
@libtuner(
    configs=addmm_get_configs(),
    key=["M", "N", "K", "stride_am", "stride_bk", "dtype"],
    strategy=["align32", "align32", "align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
)
@triton.jit(do_not_specialize=["alpha", "beta"])
def addmm_kernel_general_host_tma(
    a_desc,
    b_desc,
    i_desc,
    c_desc,
    alpha,
    beta,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_im,
    stride_in,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
    dtype: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_am = (pid_m * BLOCK_SIZE_M).to(tl.int32)
    offset_bn = (pid_n * BLOCK_SIZE_N).to(tl.int32)
    iters = tl.cdiv(K, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(iters):
        offset_ak = (k * BLOCK_SIZE_K).to(tl.int32)

        if A_ROW_MAJOR:
            a = a_desc.load([offset_am, offset_ak])
        else:
            a_t = a_desc.load([offset_ak, offset_am])
            a = tl.trans(a_t)

        if B_ROW_MAJOR:
            b = b_desc.load([offset_ak, offset_bn])
        else:
            b_t = b_desc.load([offset_bn, offset_ak])
            b = tl.trans(b_t)

        if a_desc.dtype == tl.float16 or a_desc.dtype == tl.bfloat16:
            accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)
        else:
            accumulator = tl.dot(a, b, acc=accumulator, input_precision="tf32x3")

    # Load bias
    bias = i_desc.load([offset_am, offset_bn])

    # Apply alpha and beta
    accumulator = accumulator * alpha + bias * beta
    c = accumulator.to(c_desc.dtype)
    c_desc.store([offset_am, offset_bn], c)


def general_addmm(bias, mat1, mat2, out, M, N, K, alpha=1, beta=1):
    """General addmm implementation with TMA support"""
    logger.debug(
        "GEMS ADDMM-hopper, [addmm scenario]: general, [shape info]: [-, %s, %s, %s](batch, M, N, K), "
        "[A column-major]: %s, [B column-major]: %s, [bias column-major]: %s",
        M,
        N,
        K,
        mat1.stride(0) == 1,
        mat2.stride(0) == 1,
        bias.stride(0) == 1,
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    if is_tma_compatible(bias, mat1, mat2, N, K):
        a_row_major = mat1.stride(1) == 1
        b_row_major = mat2.stride(1) == 1
        i_row_major = bias.stride(1) == 1
        dummy_block = [1, 1]
        # triton 3.5.0
        from triton.tools.tensor_descriptor import TensorDescriptor

        if a_row_major:
            a_desc = TensorDescriptor(mat1, mat1.shape, mat1.stride(), dummy_block)
        else:
            a_desc = TensorDescriptor(mat1, mat1.T.shape, mat1.T.stride(), dummy_block)
        if b_row_major:
            b_desc = TensorDescriptor(mat2, mat2.shape, mat2.stride(), dummy_block)
        else:
            b_desc = TensorDescriptor(mat2, mat2.T.shape, mat2.T.stride(), dummy_block)
        if i_row_major:
            i_desc = TensorDescriptor(bias, bias.shape, bias.stride(), dummy_block)
        else:
            i_desc = TensorDescriptor(bias, bias.T.shape, bias.T.stride(), dummy_block)
        c_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

        input_dtype = mat1.dtype
        dtype_str = str(input_dtype).split(".")[-1]

        with torch_device_fn.device(mat1.device):
            addmm_kernel_general_host_tma[grid](
                a_desc,
                b_desc,
                i_desc,
                c_desc,
                alpha,
                beta,
                M,
                N,
                K,
                mat1.stride(0),
                mat1.stride(1),
                mat2.stride(0),
                mat2.stride(1),
                bias.stride(0),
                bias.stride(1),
                out.stride(0),
                out.stride(1),
                A_ROW_MAJOR=a_row_major,
                B_ROW_MAJOR=b_row_major,
                dtype=dtype_str,
            )
    else:
        def alloc_fn(size: int, align: int, stream: Optional[int]):
            return torch.empty(size, dtype=torch.int8, device=mat1.device)

        triton.set_allocator(alloc_fn)

        with torch_device_fn.device(mat1.device):
            addmm_kernel_general[grid](
                mat1,
                mat2,
                bias,
                out,
                alpha,
                beta,
                M,
                N,
                K,
                mat1.stride(0),
                mat1.stride(1),
                mat2.stride(0),
                mat2.stride(1),
                bias.stride(0),
                bias.stride(1),
                out.stride(0),
                out.stride(1),
            )
    return out


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    """Addmm implementation with TMA optimization"""
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    # Allocate output
    out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    bias = bias.broadcast_to(out.shape)

    # Use TMA-optimized implementation
    return general_addmm(bias, mat1, mat2, out, M, N, K, alpha=alpha, beta=beta)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    """Addmm implementation with out parameter and TMA optimization"""
    assert mat1.shape[1] == mat2.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (mat1.shape[0], mat2.shape[1])
    ), "Incompatible input shape"
    M, K = mat1.shape
    _, N = mat2.shape

    if out is None:
        out = torch.empty((M, N), device=mat1.device, dtype=mat1.dtype)
    else:
        assert out.shape == (M, N), "Incompatible output shape"

    bias = bias.broadcast_to(out.shape)

    # Use TMA-optimized implementation
    return general_addmm(bias, mat1, mat2, out, M, N, K, alpha=alpha, beta=beta)
