from __future__ import annotations

import os
import subprocess

import torch

REINSTALL_COMMAND = (
    "python -m pip install --force-reinstall "
    "torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 "
    "--index-url https://download.pytorch.org/whl/cu124"
)


def _nvidia_smi() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({type(exc).__name__})"
    output = result.stdout.strip()
    return output or result.stderr.strip() or f"failed with exit code {result.returncode}"


def diagnostics() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "torch_cuda_build": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "nvidia_smi": _nvidia_smi(),
    }


def main() -> None:
    values = diagnostics()
    print("CUDA preflight:")
    for key, value in values.items():
        print(f"  {key}: {value}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "\nPyTorch cannot initialize CUDA. This experiment requires a CUDA GPU.\n"
            "The pinned, driver-compatible PyTorch build can be restored with:\n\n"
            f"  {REINSTALL_COMMAND}\n\n"
            "Then rerun `python -m sae_comp.cuda_check`. If it still fails, check "
            "`nvidia-smi` and CUDA_VISIBLE_DEVICES."
        )
    print(f"  selected_gpu: {torch.cuda.get_device_name(torch.cuda.current_device())}")


if __name__ == "__main__":
    main()
