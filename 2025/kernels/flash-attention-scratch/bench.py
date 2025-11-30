import os
import torch
import time
import math
from typing import Callable, Optional
from torch.utils.cpp_extension import load
from pathlib import Path


from flash_attn import flash_attn_func


def get_project_dir():
    return os.path.dirname(os.path.abspath(__file__))


my_flash_attn = load(
    name="my_flash_attn",
    sources=['./src/flash.cpp', './src/flash.cu'],
    extra_cuda_cflags=[
        '-O3',
        '-std=c++20',
        '--use_fast_math',
        '-Xptxas=-warn-spills',
        '-Xptxas=-warn-lmem-usage',
        '--resource-usage',
    ],
    extra_include_paths=[str(Path(get_project_dir()) / 'src/include')]
)


def benchmark_kernel(
    func: Callable,
    name: str,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    warmup: int = 32,
    iters: int = 128,
) -> None:
    """
    Benchmarks a specific attention kernel.

    Args:
        func: A lambda/partial function invoking the kernel (e.g., func()).
        name: A label for the kernel being tested.
        batch_size, num_heads, seq_len, head_dim: Dimensions for FLOP calculations.
        warmup: Number of iterations to run before timing (to warm up GPU caches).
        iters: Number of iterations to run for timing averages.
    """

    # 1. Warmup Phase
    # We run the function a few times to initialize CUDA contexts
    # and get the GPU clock speeds up.
    print(f"Benchmarking {name}...")
    try:
        for _ in range(warmup):
            func()
    except Exception as e:
        print(f"Failed during warmup for {name}: {e}")
        return

    # Synchronize to ensure all warmup kernels are finished
    torch.cuda.synchronize()

    # 2. Timing Phase
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        func()
    end_event.record()

    # Synchronize again to ensure the timer captures the full execution
    torch.cuda.synchronize()

    # Calculate elapsed time
    # elapsed_time returns milliseconds
    total_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_ms / iters
    avg_time_s = avg_time_ms / 1000.0

    # 3. Calculate FLOPs
    # Attention FLOPs ~= 4 * B * H * N^2 * D
    # 2 * (N^2 * D) for QK^T
    # 2 * (N^2 * D) for Score * V
    flops_per_iter = 4 * batch_size * num_heads * (seq_len**2) * head_dim
    tflops = flops_per_iter / avg_time_s / 1e12

    print(f"  > Avg Time: {avg_time_ms:.4f} ms")
    print(f"  > Throughput: {tflops:.2f} TFLOPS")
    print("-" * 40)


def naive_attention(q, k, v, scale):
    """
    Standard 'Manual' implementation of Attention.
    Requires storing the N x N attention matrix in memory.
    """
    # q, k, v shape: (Batch, Heads, SeqLen, Dim)

    # 1. Q * K.T -> (Batch, Heads, SeqLen, SeqLen)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # 2. Softmax
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)

    # 3. Attn * V -> (Batch, Heads, SeqLen, Dim)
    output = torch.matmul(attn_probs, v)
    return output


def run_benchmark():
    # --- Configuration ---
    BATCH_SIZE = 4
    NUM_HEADS = 16
    HEAD_DIM = 128
    SEQ_LEN = 4096  # Careful: If too large, Naive attention will OOM
    DTYPE = torch.float16  # Flash Attention usually requires fp16 or bf16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Shape: [B={BATCH_SIZE}, H={NUM_HEADS}, L={SEQ_LEN}, D={HEAD_DIM}]")
    print("-" * 40)

    # --- Data Preparation ---
    # Shape: (Batch, Heads, SeqLen, HeadDim) for Naive
    shape = (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)

    q = torch.randn(shape, device=device, dtype=DTYPE, requires_grad=False)
    k = torch.randn(shape, device=device, dtype=DTYPE, requires_grad=False)
    v = torch.randn(shape, device=device, dtype=DTYPE, requires_grad=False)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # --- 1. Test Naive Attention ---
    # We use a lambda to freeze the arguments so the benchmark function
    # just needs to call func()
    benchmark_kernel(
        func=lambda: naive_attention(q, k, v, scale),
        name="Naive Attention (Manual)",
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        seq_len=SEQ_LEN,
        head_dim=HEAD_DIM
    )

    # flash_attn_func expects shape (batch, seqlen, nheads, headdim)
    # We need to transpose from (batch, heads, seqlen, dim) to (batch, seqlen, heads, dim)
    q_flash = q.transpose(1, 2).contiguous()
    k_flash = k.transpose(1, 2).contiguous()
    v_flash = v.transpose(1, 2).contiguous()

    benchmark_kernel(
        func=lambda: flash_attn_func(
            q_flash, k_flash, v_flash, softmax_scale=scale),
        name="Flash Attention (tridao/flash-attention)",
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        seq_len=SEQ_LEN,
        head_dim=HEAD_DIM
    )

    # My Flash Attention kernel
    benchmark_kernel(
        func=lambda: my_flash_attn.forward(q, k, v),  # type: ignore
        name="My Flash Attention (kernel 1)",
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        seq_len=SEQ_LEN,
        head_dim=HEAD_DIM
    )


if __name__ == "__main__":
    run_benchmark()
    # print(get_project_dir())
