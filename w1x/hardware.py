"""Определение параметров железа (VRAM / RAM) и оценка совместимости моделей."""
import ctypes
import platform
import subprocess
import sys

from .types import HardwareInfo


def detect_hardware() -> HardwareInfo:
    """Определить доступный объём RAM и VRAM на Windows, macOS или Linux."""
    hw: HardwareInfo = {"gpu_name": "Не определена", "vram_gb": 0.0, "ram_gb": 8.0}

    if sys.platform == "win32":
        try:
            class _MEM(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("sullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MEM()
            stat.dwLength = ctypes.sizeof(_MEM)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                hw["ram_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 1)
        except Exception:
            pass

    elif sys.platform == "darwin":
        try:
            mem_raw = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL, timeout=5
            ).strip()
            if mem_raw.isdigit():
                hw["ram_gb"] = round(int(mem_raw) / (1024 ** 3), 1)
            brand = ""
            try:
                brand = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()
            except Exception:
                pass
            if platform.machine().lower() == "arm64" or "apple" in brand.lower():
                hw["gpu_name"] = "Apple Silicon (Unified Memory)"
                hw["vram_gb"] = hw["ram_gb"]
                return hw
            if brand:
                hw["gpu_name"] = brand
        except Exception:
            pass

    else:
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        hw["ram_gb"] = round(kb / (1024 ** 2), 1)
                        break
        except Exception:
            pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        if lines:
            gpu_names = []
            gpu_vrams = []
            for line in lines:
                split_parts = [p.strip() for p in line.split(",")]
                if len(split_parts) >= 2 and split_parts[1].isdigit():
                    gpu_names.append(split_parts[0])
                    gpu_vrams.append(int(split_parts[1]))

            if gpu_names and gpu_vrams:
                if len(gpu_names) == 1:
                    hw["gpu_name"] = gpu_names[0]
                    hw["vram_gb"] = round(gpu_vrams[0] / 1024, 1)
                else:
                    hw["gpu_name"] = f"{len(gpu_names)}x {gpu_names[0]}"
                    hw["vram_gb"] = round(gpu_vrams[0] / 1024, 1)
    except Exception:
        pass

    return hw


def classify_model_fit(vram_min: float, size_gb: float, hw: HardwareInfo) -> str:
    """Определить категорию нагрузки модели: green, orange, red или custom."""
    if size_gb <= 0 or vram_min <= 0:
        return "custom"

    vram_gb = hw.get("vram_gb", 0.0)
    ram_gb = hw.get("ram_gb", 8.0)

    if "Apple Silicon" in hw.get("gpu_name", ""):
        eff_unified = max(0.0, ram_gb - 2.5)
        return "green" if vram_min <= eff_unified else "red"

    eff_vram = max(0.0, vram_gb - 1.0)
    eff_total = eff_vram + max(0.0, ram_gb - 3.5)

    if eff_vram > 0 and vram_min <= eff_vram:
        return "green"
    if vram_min <= eff_total:
        return "orange"
    return "red"
