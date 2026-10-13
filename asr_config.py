"""Streaming deployment controls; environment variables can override these."""
import os
from pathlib import Path

# Keep KenLM enabled for deployment; the corpus word prior is also retained.
enable_language_model = True

# Full-recording comparisons retain the original captions with this faster path.
# Use "accuracy" for an explicit comparison with the pre-optimization decoder.
decoder_mode = "optimized"

# Recognition runs ahead of the listener's 30-second delayed playback.
recognition_mode = "blocks"
block_seconds = 4


def configured_recognition():
    mode = os.environ.get("ASR_RECOGNITION_MODE", recognition_mode).strip().lower()
    block = int(os.environ.get("ASR_BLOCK_SECONDS", block_seconds))
    if mode not in {"blocks", "rolling"} or block not in {2, 4}:
        raise ValueError("ASR_RECOGNITION_MODE must be blocks or rolling; ASR_BLOCK_SECONDS must be 2 or 4")
    return mode, block


def configured_decoder_mode() -> str:
    value = os.environ.get("ASR_DECODER_MODE", decoder_mode).strip().lower()
    if value not in {"accuracy", "optimized"}:
        raise ValueError("ASR_DECODER_MODE must be accuracy or optimized")
    return value


def language_model_enabled() -> bool:
    value = os.environ.get("ENABLE_LANGUAGE_MODEL", str(enable_language_model)).strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError("ENABLE_LANGUAGE_MODEL must be True or False")


def default_encoder_threads(cpu_budget: int) -> int:
    """Leave CPU capacity for frontend, decoder and matching on larger hosts."""
    if cpu_budget <= 2:
        return max(1, cpu_budget)
    return min(4, max(2, cpu_budget // 2))


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
