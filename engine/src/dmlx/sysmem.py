"""macOS system memory probes: totals, available memory, pressure level, swap.

Everything here is read-only and cheap enough to poll every few hundred ms.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
from dataclasses import dataclass

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

GB = 1024**3

# kern.memorystatus_vm_pressure_level values
PRESSURE_NORMAL = 1
PRESSURE_WARN = 2
PRESSURE_CRITICAL = 4
PRESSURE_NAMES = {PRESSURE_NORMAL: "normal", PRESSURE_WARN: "warn", PRESSURE_CRITICAL: "critical"}


def _sysctl_raw(name: str, size: int) -> bytes:
    buf = ctypes.create_string_buffer(size)
    length = ctypes.c_size_t(size)
    rc = _libc.sysctlbyname(name.encode(), buf, ctypes.byref(length), None, ctypes.c_size_t(0))
    if rc != 0:
        raise OSError(ctypes.get_errno(), f"sysctlbyname({name}) failed")
    return buf.raw[: length.value]


def _sysctl_int(name: str) -> int:
    raw = _sysctl_raw(name, 8)
    return int.from_bytes(raw, "little")


class _XswUsage(ctypes.Structure):
    _fields_ = [
        ("xsu_total", ctypes.c_uint64),
        ("xsu_avail", ctypes.c_uint64),
        ("xsu_used", ctypes.c_uint64),
        ("xsu_pagesize", ctypes.c_uint32),
        ("xsu_encrypted", ctypes.c_bool),
    ]


class _VMStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


_HOST_VM_INFO64 = 4
_HOST_VM_INFO64_COUNT = ctypes.sizeof(_VMStatistics64) // ctypes.sizeof(ctypes.c_int)
_libc.mach_host_self.restype = ctypes.c_uint32
_libc.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]


@dataclass(frozen=True)
class MemorySnapshot:
    total: int
    available: int  # free + inactive + purgeable + speculative (reclaimable without swap)
    wired: int
    compressed: int
    swap_used: int
    pressure: int  # PRESSURE_*
    swapins: int = 0  # cumulative bytes read back from swap (thrash signal)
    swapouts: int = 0
    taken: float = 0.0

    @property
    def pressure_name(self) -> str:
        return PRESSURE_NAMES.get(self.pressure, str(self.pressure))

    def as_dict(self) -> dict:
        return {
            "total_gb": round(self.total / GB, 2),
            "available_gb": round(self.available / GB, 2),
            "wired_gb": round(self.wired / GB, 2),
            "compressed_gb": round(self.compressed / GB, 2),
            "swap_used_gb": round(self.swap_used / GB, 2),
            "pressure": self.pressure_name,
        }


def total_memory() -> int:
    override = os.environ.get("DMLX_EMULATE_TOTAL_GB")
    if override:
        return int(float(override) * GB)
    return _sysctl_int("hw.memsize")


def page_size() -> int:
    return _sysctl_int("hw.pagesize")


def pressure_level() -> int:
    try:
        return _sysctl_int("kern.memorystatus_vm_pressure_level")
    except OSError:
        return PRESSURE_NORMAL


def swap_used() -> int:
    raw = _sysctl_raw("vm.swapusage", ctypes.sizeof(_XswUsage))
    usage = _XswUsage.from_buffer_copy(raw.ljust(ctypes.sizeof(_XswUsage), b"\0"))
    return int(usage.xsu_used)


def _vm_stats() -> _VMStatistics64:
    stats = _VMStatistics64()
    count = ctypes.c_uint32(_HOST_VM_INFO64_COUNT)
    rc = _libc.host_statistics64(_libc.mach_host_self(), _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count))
    if rc != 0:
        raise OSError(rc, "host_statistics64 failed")
    return stats


def snapshot() -> MemorySnapshot:
    ps = page_size()
    vm = _vm_stats()
    real_total = _sysctl_int("hw.memsize")
    total = total_memory()
    # Reclaimable without swapping anyone: free pages plus file cache (external pages can be dropped).
    # Other apps' inactive anonymous memory is NOT counted: taking it forces compression or swap.
    available = (vm.free_count + vm.speculative_count + vm.purgeable_count + min(vm.inactive_count, vm.external_page_count)) * ps
    if os.environ.get("DMLX_EMULATE_STRICT") == "1":
        # Strict emulation: the memory we pretend not to have is unavailable.
        available = max(0, available - (real_total - total))
    return MemorySnapshot(
        total=total,
        available=available,
        wired=vm.wire_count * ps,
        compressed=vm.compressor_page_count * ps,
        swap_used=swap_used(),
        pressure=pressure_level(),
        swapins=vm.swapins * ps,
        swapouts=vm.swapouts * ps,
        taken=time.monotonic(),
    )


def os_reserve(total: int) -> int:
    """Reclaimable memory we never plan to take, so macOS and foreground apps keep room to grow.

    Applied on top of `available`, which already excludes everything other apps are using.
    """
    return int(min(max(0.08 * total, 1.5 * GB), 4 * GB))
