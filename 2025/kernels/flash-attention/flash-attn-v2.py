import torch

import triton
import triton.language as tl

from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _attn_fwd(
    sm_scale,
    BATCH, N_HEADS, N_CTX, HEAD_DIM: tl.constexpr,
    Q, K, V, O, M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = tl.float32
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    start_m = tl.program_id(0)

    # Which batch and head
    off_batch_head = tl.program_id(1)
    # Which batch
    off_batch = off_batch_head // N_HEADS
    # Which head
    off_head = off_batch_head % N_HEADS

    y_dim = BATCH * N_HEADS * N_CTX
    desc_q = tl.make_tensor_descriptor(Q, shape=[y_dim, HEAD_DIM],
                                       strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(V, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(K, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(O, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])

    # Which (batch, head)
    offset_y = off_batch * (N_HEADS * N_CTX) + off_head * (N_CTX)
    qo_offset_y = offset_y + start_m * BLOCK_M
    # Initialize offsets
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=dtype) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=dtype) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=dtype)

    # Load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # Load Q: it will stay in SRAM
    q = desc_q.load([qo_offset_y, 0])

    offsetk_y = offset_y
    offsetv_y = offset_y

    # loop over K, V and update accumulator
    for start_n in tl.range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # compute qk
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)

        # Apply scale to qk first, then compute max
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)

        # compute correction factor
        l_ij = tl.sum(p, 1)

        # Update output accumulator
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]

        # prepare p and v for the dot
        v = desc_v.load([offsetv_y, 0])
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    m_ptrs = M + off_batch_head * N_CTX + off_m
    tl.store(m_ptrs, m_i)

    desc_o.store([qo_offset_y, 0], acc.to(dtype))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k, v, sm_scale):
        o = torch.empty_like(q)

        print(q.shape[0], q.shape[1], q.shape[2], q.shape[3])

        BATCH = q.shape[0]
        N_HEADS = q.shape[1]
        N_CTX = q.shape[2]
        HEAD_DIM = q.shape[3]

        BLOCK_M = 64
        BLOCK_N = 64

        def grid(META):
            return (triton.cdiv(N_CTX, BLOCK_M), BATCH * N_HEADS, 1)

        _attn_fwd[grid](
            sm_scale,
            BATCH, N_HEADS, N_CTX, HEAD_DIM,
            q, k, v, o,
            torch.empty((BATCH * N_HEADS, N_CTX),
                        device=q.device, dtype=torch.float32),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return o


def ref_attn(q, k, v, sm_scale=1):
    SEQLEN = q.shape[-2]
    M = torch.tril(torch.ones((SEQLEN, SEQLEN), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1)
    ref_out = torch.matmul(p, v)
    return ref_out


attention = _attention.apply

BATCH = 4
N_HEADS = 32
N_CTX = 1024
HEAD_DIM = 64

dtype = torch.float32
q = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM),
                device=DEVICE, dtype=dtype, requires_grad=True)
k = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM),
                device=DEVICE, dtype=dtype, requires_grad=True)
v = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM),
                device=DEVICE, dtype=dtype, requires_grad=True)

sm_scale = 0.5

tri_out = attention(q, k, v, sm_scale)
ref_out = ref_attn(q, k, v, sm_scale)


# compare
assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
