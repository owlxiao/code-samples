import math
import torch

from my_flash_attn import flash_attn_func
import torch.utils.benchmark as benchmark


def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper,
                 "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def time_fwd(func, *args, **kwargs):
    time = benchmark_forward(func, *args, **kwargs)
    return time[1].mean


def flops(batch, seqlen, headdim, nheads, causal):
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


batch_size = 4
headdim = 128
nheads = 8
seqlen = 4096

q = torch.randn(batch_size, seqlen, nheads, headdim,
                dtype=torch.float16, device="cuda")
k = torch.randn(batch_size, seqlen, nheads, headdim,
                dtype=torch.float16, device="cuda")
v = torch.randn(batch_size, seqlen, nheads, headdim,
                dtype=torch.float16, device="cuda")

time = time_fwd(flash_attn_func, q, k, v, causal=False)

print(
    f"### headdim={headdim}, nheads={nheads}, seqlen={seqlen}, batch_size={batch_size} ###")
speed = efficiency(flops(batch_size, seqlen, headdim, nheads, False), time)
print(f"Speed: {speed:.2f} TFLOPS")
