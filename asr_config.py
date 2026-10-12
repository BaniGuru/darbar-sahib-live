"""Streaming deployment controls; environment variables can override these."""
import os
from pathlib import Path

# Disable the optional KenLM n-gram; retain the existing corpus word prior.
enable_language_model = True


def language_model_enabled() -> bool:
    value = os.environ.get("ENABLE_LANGUAGE_MODEL", str(enable_language_model)).strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError("ENABLE_LANGUAGE_MODEL must be True or False")


def available_cpu_count() -> int:
    """Respect container quota and affinity, rather than host CPU count."""
    limits = [os.cpu_count() or 1]
    try:
        limits.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits.append(max(1, int(quota) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                limits.append(max(1, quota // period))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    return max(1, min(limits))
