import math
import torch
import pytest
from einops import repeat, rearrange
from my_flash_attn import flash_attn_func


def attention_ref(q, k, v, attn_mask=None, key_padding_mask=None, attn_bias=None,
                  dropout_p=0.0, dropout_mask=None, causal=False, window_size=(-1, -1),
                  softcap=0.0):
    """
    Reference implementation of attention mechanism.
    
    Args:
        q: Query tensor of shape (batch, seqlen, nheads, d)
        k: Key tensor of shape (batch, seqlen, nheads, d)
        v: Value tensor of shape (batch, seqlen, nheads, d)
    
    Returns:
        out: Output tensor of shape (batch, seqlen, nheads, d)
        attn: Attention weights of shape (batch, nheads, seqlen, seqlen)
    """
    # Rearrange to (batch, nheads, seqlen, d)
    q = rearrange(q, 'b s h d -> b h s d')
    k = rearrange(k, 'b s h d -> b h s d')
    v = rearrange(v, 'b s h d -> b h s d')

    # Compute attention scores
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    # Apply attention weights
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)

    # Rearrange back to (batch, seqlen, nheads, d)
    out = rearrange(out, 'b h s d -> b s h d')

    return out


@pytest.mark.parametrize("seqlen", [512, 1024, 2048])
@pytest.mark.parametrize("d", [128])
def test_flash_attn_output(seqlen, d):
    device = 'cuda'
    batch_size = 4
    nheads = 8
    d = 128

    q = torch.randn(batch_size, seqlen, nheads, d,
                    device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, d,
                    device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, d,
                    device=device, dtype=torch.float16, requires_grad=True)

    out = flash_attn_func(q, k, v)
    out_ref = attention_ref(q, k, v)

   # Compute differences
    max_diff = (out - out_ref).abs().max().item()
    mean_diff = (out - out_ref).abs().mean().item()

    print(f"Output max diff: {max_diff:.6e}")
    print(f"Output mean diff: {mean_diff:.6e}")

    # Check outputs are close (fp16 tolerance)
    rtol = 1e-2
    atol = 1e-2

    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol), \
        f"Outputs differ too much: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"


if __name__ == "__main__":
    test_flash_attn_output(512, 128)
