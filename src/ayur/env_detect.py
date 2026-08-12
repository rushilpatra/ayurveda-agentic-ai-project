"""Hardware/software detection for local Apple-silicon execution.

Detects at runtime; nothing about this machine is hard-coded. Emits a JSON record
used to pick a safe model tier and to stamp every results table with the
environment it was produced on.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _sh(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sysctl(key: str) -> str | None:
    return _sh(["sysctl", "-n", key])


def _int_sysctl(key: str) -> int | None:
    v = _sysctl(key)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


@dataclass
class Environment:
    os_name: str
    os_version: str | None
    build: str | None
    arch: str
    chip: str | None
    cpu_logical: int | None
    cpu_performance: int | None
    cpu_efficiency: int | None
    memory_gb: float | None
    disk_free_gb: float | None
    python_version: str
    python_executable: str
    is_apple_silicon: bool
    mlx_available: bool = False
    mlx_version: str | None = None
    mlx_metal: bool = False
    torch_available: bool = False
    torch_version: str | None = None
    torch_mps: bool = False
    warnings: list[str] = field(default_factory=list)

    # ---- derived policy ----
    @property
    def model_tier(self) -> str:
        """Safe MLX model tier for the detected unified memory.

        Thresholds follow the project brief. Deliberately conservative: unified
        memory is shared with the OS and every other process.
        """
        gb = self.memory_gb or 0
        if gb >= 24:
            return "qwen3-8b-4bit"
        if gb >= 16:
            return "qwen3-4b-4bit"
        if gb >= 8:
            return "qwen3-1.7b-4bit"
        return "cpu-only"

    @property
    def recommended_threads(self) -> int:
        """Leave headroom; do not saturate the machine."""
        n = self.cpu_performance or self.cpu_logical or 4
        return max(1, min(n, 8))


MODEL_REPOS = {
    "qwen3-8b-4bit": "mlx-community/Qwen3-8B-4bit",
    "qwen3-4b-4bit": "mlx-community/Qwen3-4B-4bit",
    "qwen3-1.7b-4bit": "mlx-community/Qwen3-1.7B-4bit",
    "cpu-only": None,
}


def detect() -> Environment:
    warnings: list[str] = []

    mem_bytes = _int_sysctl("hw.memsize")
    memory_gb = round(mem_bytes / 1024**3, 1) if mem_bytes else None

    disk_free_gb = None
    try:
        usage = shutil.disk_usage(Path.home())
        disk_free_gb = round(usage.free / 1024**3, 1)
    except Exception:
        warnings.append("could not determine free disk space")

    machine = platform.machine()
    is_apple_silicon = sys.platform == "darwin" and machine == "arm64"

    env = Environment(
        os_name=platform.system(),
        os_version=_sh(["sw_vers", "-productVersion"]) or platform.release(),
        build=_sh(["sw_vers", "-buildVersion"]),
        arch=machine,
        chip=_sysctl("machdep.cpu.brand_string"),
        cpu_logical=_int_sysctl("hw.ncpu"),
        cpu_performance=_int_sysctl("hw.perflevel0.logicalcpu"),
        cpu_efficiency=_int_sysctl("hw.perflevel1.logicalcpu"),
        memory_gb=memory_gb,
        disk_free_gb=disk_free_gb,
        python_version=platform.python_version(),
        python_executable=sys.executable,
        is_apple_silicon=is_apple_silicon,
        warnings=warnings,
    )

    if not is_apple_silicon:
        env.warnings.append(
            f"not Apple silicon (platform={sys.platform}, machine={machine}); "
            "MLX paths will be unavailable"
        )

    try:
        import mlx.core as mx

        env.mlx_available = True
        try:
            import mlx

            env.mlx_version = getattr(mlx, "__version__", None)
        except Exception:
            pass
        try:
            env.mlx_metal = mx.metal.is_available()
        except Exception:
            # newer mlx moved this; absence is not fatal
            env.mlx_metal = bool(is_apple_silicon)
    except ImportError:
        env.warnings.append("mlx not installed - run `make setup`")

    try:
        import torch

        env.torch_available = True
        env.torch_version = torch.__version__
        env.torch_mps = bool(torch.backends.mps.is_available())
    except ImportError:
        env.warnings.append("torch not installed (only needed for sentence-transformers)")
    except Exception as e:  # pragma: no cover
        env.warnings.append(f"torch present but unusable: {e}")

    if memory_gb and memory_gb < 16:
        env.warnings.append(
            f"{memory_gb} GB unified memory is below the 16 GB the pipeline targets; "
            "use the smallest model tier and reduce batch sizes"
        )
    if disk_free_gb is not None and disk_free_gb < 20:
        env.warnings.append(
            f"only {disk_free_gb} GB free; a 4-bit 8B model needs ~5 GB plus caches"
        )

    return env


def as_dict(env: Environment) -> dict:
    d = asdict(env)
    d["model_tier"] = env.model_tier
    d["model_repo"] = MODEL_REPOS[env.model_tier]
    d["recommended_threads"] = env.recommended_threads
    return d


def main() -> int:
    env = detect()
    d = as_dict(env)

    out = Path("results/environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2))

    print("=" * 64)
    print("ENVIRONMENT")
    print("=" * 64)
    print(f"  OS               {env.os_name} {env.os_version} ({env.build})")
    print(f"  Chip             {env.chip}  [{env.arch}]")
    print(f"  CPU              {env.cpu_logical} logical "
          f"({env.cpu_performance}P / {env.cpu_efficiency}E)")
    print(f"  Unified memory   {env.memory_gb} GB")
    print(f"  Disk free        {env.disk_free_gb} GB")
    print(f"  Python           {env.python_version}")
    print(f"  MLX              {'yes' if env.mlx_available else 'NO'}"
          f"  (metal={env.mlx_metal}, version={env.mlx_version})")
    print(f"  PyTorch          {'yes' if env.torch_available else 'NO'}"
          f"  (mps={env.torch_mps}, version={env.torch_version})")
    print("-" * 64)
    print(f"  -> model tier    {env.model_tier}")
    print(f"  -> model repo    {MODEL_REPOS[env.model_tier]}")
    print(f"  -> threads       {env.recommended_threads}")
    if env.warnings:
        print("-" * 64)
        for w in env.warnings:
            print(f"  ! {w}")
    print("=" * 64)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
