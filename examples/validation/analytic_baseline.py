#!/usr/bin/env python3
# Analytical no-contention baseline for the DP transformer step (mirrors train.cu dims).

import argparse, math

H, T, FF = 4096, 2048, 4 * 4096
GRAD_ELEMS = 12 * H * H
TYPE_SIZE = 4

KERNELS = [
    ("layernorm",    T * H,     0,    8, False),
    ("qkv_proj",     T * 3 * H, 8192, 20, True),
    ("attn_scores",  T * T,     8192, 20, True),
    ("softmax_rows", T * T,     0,    8, False),
    ("attn_av",      T * H,     4096, 20, True),
    ("attn_out",     T * H,     8192, 20, True),
    ("layernorm2",   T * H,     0,    8, False),
    ("mlp_up",       T * FF,    8192, 20, True),
    ("gelu",         T * FF,    0,    8, False),
    ("mlp_down",     T * H,     32768, 20, True),
]
BLOCK = 256


def wave_penalty(threads, sm_count, max_threads_per_sm=2048, max_blocks_per_sm=32):
    """ceil(blocks/concurrent)/(blocks/concurrent); 1.0 when sm_count==0."""
    if sm_count <= 0:
        return 1.0
    blocks = math.ceil(threads / BLOCK)
    bps = min(max_threads_per_sm // BLOCK, max_blocks_per_sm) or 1
    concurrent = sm_count * bps
    ideal = blocks / concurrent
    return math.ceil(ideal) / ideal if ideal > 0 else 1.0


def kernel_time(threads, flops_pt, bytes_pt, is_tensor, a):
    peak = a.tensor_peak if is_tensor else a.fp32_peak
    compute = (flops_pt * threads) / peak if peak > 0 else 0.0
    mem = (bytes_pt * threads) / a.mem_bw if a.mem_bw > 0 else 0.0
    return a.launch_overhead + max(compute, mem) * wave_penalty(threads, a.sm_count)


def layer_compute(a):
    return sum(kernel_time(t, f, b, tc, a) for (_, t, f, b, tc) in KERNELS)


def step_compute(a):
    # forward L + backward 2L layer-passes.
    return (a.layers + 2 * a.layers) * layer_compute(a)


def allreduce_time(nbytes, n, a, alg):
    """Closed-form all-reduce wall time (ring or recursive-doubling), no contention."""
    if n <= 1:
        return 0.0
    bw_term = 2.0 * (n - 1) / n * nbytes / a.link_bw
    if alg == "ring":
        steps = 2 * (n - 1)
    else:
        steps = 2 * math.ceil(math.log2(n))
    return bw_term + steps * a.link_lat


def step_time(a, alg):
    compute = step_compute(a)
    if a.nranks <= 1:
        return compute, compute, 0.0
    grad_bytes = GRAD_ELEMS * TYPE_SIZE
    comms = a.layers * allreduce_time(grad_bytes, a.nranks, a, alg)
    backward = 2 * a.layers * layer_compute(a)
    if a.overlap:
        total = compute + max(0.0, comms - backward)
    else:
        total = compute + comms
    return total, compute, comms


def main():
    p = argparse.ArgumentParser(description="Analytical no-contention baseline for the DP transformer step")
    p.add_argument("--nranks", type=int, default=8)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--alg", choices=["ring", "recdouble"], default="recdouble")
    p.add_argument("--fp32-peak", type=float, default=1.95e13)
    p.add_argument("--tensor-peak", type=float, default=3.12e14)
    p.add_argument("--mem-bw", type=float, default=2000e9)
    p.add_argument("--link-bw", type=float, default=100e9)
    p.add_argument("--link-lat", type=float, default=100e-9)
    p.add_argument("--launch-overhead", type=float, default=2e-6)
    p.add_argument("--sm-count", type=int, default=108)
    p.add_argument("--overlap", type=int, default=1)
    a = p.parse_args()
    a.overlap = bool(a.overlap)

    total, compute, comms = step_time(a, a.alg)
    print(f"# analytic baseline (no contention): N={a.nranks} alg={a.alg} "
          f"layers={a.layers} link_bw={a.link_bw/1e9:g}GB/s overlap={int(a.overlap)}")
    print(f"compute_ms={compute*1e3:.4f}")
    print(f"comms_ms={comms*1e3:.4f}")
    print(f"step_ms={total*1e3:.4f}")


if __name__ == "__main__":
    main()
