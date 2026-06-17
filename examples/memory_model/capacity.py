#!/usr/bin/env python3
# Closed-form per-rank GPU memory model (weights, grads, optimizer, activations/KV).

import argparse

GB = 1e9

# H, L (transformer layers), heads, V (vocab). FF assumed 4*H (-> 12*H^2/layer).
PRESETS = {
    "demo": dict(H=4096,  L=8,  heads=32, V=50304),
    "7b":   dict(H=4096,  L=32, heads=32, V=32000),
    "70b":  dict(H=8192,  L=80, heads=64, V=32000),
    "175b": dict(H=12288, L=96, heads=96, V=50257),
}
DEVICE_GB = {"a100-40": 40, "a100-80": 80, "h100-80": 80, "h200-141": 141}


def param_count(H, L, V):
    return L * 12 * H * H + 2 * V * H


def state_bytes(dtype_size, optimizer, mixed):
    """Per-param (weight, grad, optimizer) bytes, split so ZeRO can partition."""
    if mixed:
        weight, grad, master = dtype_size, dtype_size, 4
    else:
        weight, grad, master = 4, 4, 0
    if optimizer == "adam":
        opt = master + 4 + 4
    elif optimizer == "sgd":
        opt = master + 4
    else:
        opt = master
    return weight, grad, opt


def training_footprint(a):
    P = param_count(a.H, a.L, a.V)
    shard = a.tp * a.pp
    p_rank = P / shard
    w, g, o = state_bytes(a.dtype, a.optimizer, a.mixed)
    dp = max(a.dp, 1)
    # ZeRO shards optimizer(1), grad(2), weight(3) over the DP group.
    wd = dp if a.zero >= 3 else 1
    gd = dp if a.zero >= 2 else 1
    od = dp if a.zero >= 1 else 1
    weights = p_rank * w / wd
    grads   = p_rank * g / gd
    optim   = p_rank * o / od
    transient = (12 * a.H * a.H) * w if a.zero >= 3 else 0.0
    act_layer = a.act_factor * a.seq * a.micro_batch * a.H * a.dtype / a.tp
    activ = act_layer * a.L
    return {"weights": weights, "grads": grads, "optimizer": optim,
            "activations": activ, "zero3_transient": transient, "_params": P}


def inference_footprint(a):
    P = param_count(a.H, a.L, a.V)
    weights = P / (a.tp * a.pp) * a.dtype
    layers_rank = a.L / a.pp
    h_kv = a.H / a.gqa
    kv = 2 * layers_rank * a.seq * h_kv * a.dtype * a.batch / a.tp
    return {"weights": weights, "kv_cache": kv, "_params": P}


def report(a):
    foot = inference_footprint(a) if a.infer else training_footprint(a)
    total = sum(v for k, v in foot.items() if not k.startswith("_"))
    cap = a.gpu_mem * GB
    mode = "inference" if a.infer else "training"
    print(f"# {mode}: {a.model} (~{foot['_params']/1e9:.1f}B params) "
          f"DP{a.dp} TP{a.tp} PP{a.pp} ZeRO{a.zero} "
          f"dtype={a.dtype}B seq={a.seq} on {a.device}({a.gpu_mem:g}GB)")
    for k, v in foot.items():
        if not k.startswith("_"):
            print(f"  {k:16s} {v/GB:8.2f} GB")
    verdict = "FITS" if total <= cap else "OOM"
    print(f"  {'TOTAL':16s} {total/GB:8.2f} GB / {a.gpu_mem:g} GB  "
          f"({100*total/cap:.0f}%)  -> {verdict}")
    return total <= cap


def sweep(a):
    """Frontier: which DP/TP/PP fit, for a fixed GPU budget N (powers of two)."""
    N = a.gpus
    print(f"# frontier: {a.model} (~{param_count(a.H,a.L,a.V)/1e9:.1f}B) on N={N} x "
          f"{a.device}({a.gpu_mem:g}GB), ZeRO{a.zero}, {'infer' if a.infer else 'train'}")
    print(f"  {'DPxPPxTP':14s} {'total_GB':>9s}  verdict")
    best = None
    tp = 1
    while tp <= N:
        pp = 1
        while tp * pp <= N:
            if N % (tp * pp) == 0 and a.L % pp == 0:
                a.dp, a.tp, a.pp = N // (tp * pp), tp, pp
                foot = inference_footprint(a) if a.infer else training_footprint(a)
                tot = sum(v for k, v in foot.items() if not k.startswith("_"))
                ok = tot <= a.gpu_mem * GB
                print(f"  DP{a.dp}xPP{pp}xTP{tp:<6d} {tot/GB:9.1f}  "
                      f"{'FITS' if ok else 'OOM'}")
                if ok and (best is None or tot < best[1]):
                    best = (f"DP{a.dp}xPP{pp}xTP{tp}", tot)
            pp *= 2
        tp *= 2
    print(f"  -> memory-optimal that fits: {best[0]} ({best[1]/GB:.1f} GB)"
          if best else "  -> NONE fit at this budget")


def main():
    p = argparse.ArgumentParser(description="Per-rank GPU memory / feasibility model")
    p.add_argument("--model", choices=list(PRESETS), default="demo")
    p.add_argument("--H", type=int); p.add_argument("--L", type=int)
    p.add_argument("--heads", type=int); p.add_argument("--V", type=int)
    p.add_argument("--no-embed", action="store_true", help="layer params only (demo cross-check)")
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--zero", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--dtype", type=int, choices=[2, 4], default=2, help="bytes/elem")
    p.add_argument("--mixed", type=int, default=1, help="mixed precision (fp32 master)")
    p.add_argument("--optimizer", choices=["adam", "sgd", "none"], default="adam")
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--act-factor", type=float)
    p.add_argument("--checkpoint", choices=["none", "selective", "full"], default="selective")
    p.add_argument("--infer", action="store_true", help="inference: weights + KV cache")
    p.add_argument("--batch", type=int, default=1, help="inference batch size")
    p.add_argument("--gqa", type=int, default=1, help="KV-head reduction (GQA)")
    p.add_argument("--device", choices=list(DEVICE_GB), default="a100-80")
    p.add_argument("--gpu-mem", type=float, help="override capacity, GB (decimal)")
    p.add_argument("--gpus", type=int, help="sweep mode: GPU budget N")
    a = p.parse_args()

    d = PRESETS[a.model]
    for k in ("H", "L", "heads", "V"):
        if getattr(a, k) is None:
            setattr(a, k, d[k])
    a.mixed = bool(a.mixed)
    if a.no_embed:
        a.V = 0
    if a.gpu_mem is None:
        a.gpu_mem = DEVICE_GB[a.device]
    if a.act_factor is None:
        a.act_factor = {"none": 17.0, "selective": 5.0, "full": 2.0}[a.checkpoint]

    if a.gpus:
        sweep(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
