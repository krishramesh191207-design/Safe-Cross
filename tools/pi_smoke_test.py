"""Run a minimal ExecuTorch forward pass for both SafeCross Pi models."""
import sys
import time
from pathlib import Path


def main():
    import torch
    from executorch.runtime import Runtime

    root = Path(__file__).resolve().parents[1]
    device_path = Path("/proc/device-tree/model")
    if device_path.exists():
        device = device_path.read_bytes().rstrip(b"\x00").decode(errors="replace")
        print(f"Pi model: {device}")
    else:
        print("Pi model: unavailable")

    dummy = torch.rand(1, 3, 320, 320)
    for relative_path in ("models/ssdlite_int8.pte", "models/ssdlite_fp32.pte"):
        model_path = root / relative_path
        try:
            method = Runtime.get().load_program(str(model_path)).load_method("forward")
            start = time.perf_counter()
            outputs = method.execute([dummy])
            elapsed_ms = (time.perf_counter() - start) * 1000
            shapes = [tuple(output.shape) for output in outputs]
            print(f"{relative_path}: output shapes={shapes}, time={elapsed_ms:.2f} ms")
        except Exception as exc:
            raise RuntimeError(f"{relative_path} failed: {exc}") from exc


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pi smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
