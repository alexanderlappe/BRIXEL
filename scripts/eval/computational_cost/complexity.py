import argparse
import csv
import os
import time
import gc
from pathlib import Path
from typing import Iterable

import torch
from torchvision.transforms.functional import resize
from tqdm import tqdm

from modeling.dinov3.upsampling.eval.segmentation.ade20k_linear_probe import build_backbone
from modeling.dinov3.upsampling.transforms import GetLargeView
from modeling.dinov3.upsampling.utils.Dataset import ImageDataset


def _fmt_shape(t):
    return "x".join(map(str, t)) if isinstance(t, tuple) else ""

def save_metrics_csv_stdlib(results: dict, path: str):
    # Define a stable column order
    columns: Iterable[str] = [
        "model",
        "device", "amp", "wall_time_s",
        "flops", "gflops",
        "params", "params_m",
        "peak_memory_bytes", "peak_mem_mib",
        "output_shape",
    ]


    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for name, m in results.items():
            flops = m.get("flops")
            params = m.get("params")
            peak = m.get("peak_memory_bytes")

            row = [
                name,
                m.get("device", ""),
                m.get("amp", ""),
                m.get("wall_time_s", ""),
                flops if flops is not None else "",
                (flops / 1e9) if isinstance(flops, (int, float)) else "",
                params if params is not None else "",
                (params / 1e6) if isinstance(params, (int, float)) else "",
                peak if peak is not None else "",
                (peak / (1024**2)) if isinstance(peak, (int, float)) else "",
                _fmt_shape(m.get("output_shape")),
            ]
            w.writerow(row)
def bytes_human(n):
    for unit in ("B","KiB","MiB","GiB","TiB"):
        if n < 1024: return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"

def available_vram():
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free, total
    return None, None

def dtype_nbytes(dtype: torch.dtype) -> int:
    return {
        torch.float32: 4, torch.float: 4,
        torch.float16: 2, torch.half: 2,
        torch.bfloat16: 2,
        torch.int8: 1, torch.uint8: 1,
        torch.int32: 4, torch.int64: 8,
    }.get(dtype, 4)

def preflight_param_bytes(model, param_dtype=torch.float32):
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    return (n_params + n_buffers) * dtype_nbytes(param_dtype)

@torch.inference_mode()
def measure_flops_cpu(model, x_cpu):
    import copy
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn.jit_handles import addmm_flop_jit, matmul_flop_jit, bmm_flop_jit
    from thop import profile
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    flops = None
    try:
        def sdpa_flop_jit(inputs_, outputs_):
            q, k, v = inputs_[:3]
            B,H,N,Dh = q.shape
            return int(4*B*H*N*N*Dh)  # qk^T + attn*v
        fca = FlopCountAnalysis(model_cpu, x_cpu)
        fca.set_op_handle("aten::addmm", addmm_flop_jit)
        fca.set_op_handle("aten::matmul", matmul_flop_jit)
        fca.set_op_handle("aten::bmm",    bmm_flop_jit)
        fca.set_op_handle("aten::scaled_dot_product_attention", sdpa_flop_jit)
        flops = int(fca.total())
    except Exception:
        try:
            macs, _ = profile(model_cpu, inputs=(x_cpu,), verbose=False)
            flops = int(macs * 2)
        except Exception:
            flops = None
    finally:
        del model_cpu
        gc.collect()
    return flops


@torch.inference_mode()
def safe_measure_one(model, inputs, use_autocast=False, amp_dtype=torch.float16, device="cuda"):
    try:
        dev = torch.device(device if (device=="cpu" or torch.cuda.is_available()) else "cpu")
        model = model.to(dev).eval()

        # quick VRAM preflight
        param_bytes = preflight_param_bytes(
            model,
            next(model.parameters(), torch.tensor([], dtype=torch.float32)).dtype
            if any(True for _ in model.parameters()) else torch.float32
        )
        if dev.type == "cuda":
            free, _ = available_vram()
            if free is not None and param_bytes * 1.3 > free:
                return {"error": f"Preflight skip: params {bytes_human(param_bytes)} > {bytes_human(free)} free VRAM"}

        x = inputs.to(dev, non_blocking=True)

        # tiny warmup (0–1 iters is plenty)
        for _ in range(1):
            with (torch.autocast(device_type="cuda", dtype=amp_dtype) if (use_autocast and dev.type=="cuda") else torch.no_grad()):
                _ = model(x)

        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with (torch.autocast(device_type="cuda", dtype=amp_dtype) if (use_autocast and dev.type=="cuda") else torch.no_grad()):
            out = model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        peak_mem = torch.cuda.max_memory_allocated() if dev.type == "cuda" else None
        n_params = sum(p.numel() for p in model.parameters())

        return {
            "flops": None,  # filled in by caller
            "params": n_params,
            "peak_memory_bytes": peak_mem,
            "wall_time_s": t1 - t0,
            "output_shape": tuple(out.shape) if hasattr(out, "shape") else None,
            "amp": bool(use_autocast),
            "device": str(dev),
        }
    except torch.cuda.OutOfMemoryError:
        return {"error": "CUDA OOM during allocation/forward"}
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def get_args():
    # TODO Remove the defaults and set necessary arguments properly
    ap = argparse.ArgumentParser("Complexity")
    ap.add_argument("--img-dir", type=str, required=False, default="/media/alex/DataSSD/alex/Dense/images/00000/")
    ap.add_argument("--config-path", type=str, required=False, default="/home/alex/PycharmProjects/TheSecretOne/modeling/dinov3/upsampling/eval/eval_config.json", help="Path to model config file")
    ap.add_argument("--img-size", type=int, required=False, default=256)

    ap.add_argument("--save-dir", type=str, default="/media/alex/DataSSD/alex/Dense/eval/results/complexity")
    return ap.parse_args()


def main():
    args = get_args()

    backbone_names = ['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vith16plus',
                      'dinov3_vits16_adapter', 'dinov3_vitb16_adapter', 'dinov3_vitl16_adapter', 'dinov3_vith16plus_adapter']

    if args.img_size > 512:   # don't need adapters at these sizes
        backbone_names = ['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vith16plus']


    results = {}

    print('Probing complexity...')
    for backbone_name in tqdm(backbone_names):
        model, _ = build_backbone(backbone_name, args.config_path)

        if "adapter" in backbone_name:
            patch_size = model.adapter.backbone.patch_size
        else:
            patch_size = model.patch_size
        transform = GetLargeView(patch_size=patch_size)

        valset = ImageDataset(args.img_dir)

        input = valset[0].unsqueeze(0)   # just take some random image
        input = transform(input)
        input = resize(input, size=args.img_size)

        model_cpu = model.to('cpu')
        x_cpu = input.to("cpu")

        # Phase A: FLOPs on CPU
        flops = measure_flops_cpu(model_cpu, x_cpu)

        # Phase B: timed/peak-mem on GPU (AMP optional)
        metrics = safe_measure_one(
            model,  # re-use same instance
            input,  # original tensor; safe_measure_one moves it
            use_autocast=True,  # match your “normal” mixed-precision
            amp_dtype=torch.float16,  # or bfloat16 if that’s your norm
            device="cuda"
        )

        metrics["flops"] = flops
        results[backbone_name] = metrics

        results[backbone_name] = metrics

        print(f"{backbone_name}: {metrics}")
        print('-------------------------------')

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(args.save_dir, "complexity_" + str(args.img_size) + ".csv")
    save_metrics_csv_stdlib(results, save_path)


if __name__ == "__main__":
    main()
