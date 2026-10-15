"""Shared, file-change-aware lookup of the raagi heard at an India timestamp."""
from datetime import datetime
import json
from pathlib import Path
import re
from threading import Lock
from zoneinfo import ZoneInfo


class DutySchedule:
    def __init__(self, path: Path):
        self.path = path
        self._signature = None
        self._duties = {}
        self._lock = Lock()

    def _load(self):
        try:
            stat = self.path.stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except OSError:
            self._signature = None
            self._duties = {}
            return
        if signature == self._signature:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            duties = data.get("duties") if isinstance(data, dict) else None
            self._duties = duties if isinstance(duties, dict) else {}
        except (OSError, ValueError):
            self._duties = {}
        self._signature = signature

    @staticmethod
    def _minute(value):
        if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("Invalid duty clock")
        hour, minute = map(int, value.split(":"))
        return hour*60+minute

    def at(self, timestamp: float):
        now = datetime.fromtimestamp(timestamp, ZoneInfo("Asia/Kolkata"))
        with self._lock:
            self._load()
            entries = self._duties.get(now.strftime("%d-%m-%Y"), [])
            if not isinstance(entries, list):
                return None
            try:
                ranges = sorted((self._minute(row["start_time"]),
                                 1440 if row["end_time"] == "" else self._minute(row["end_time"]),
                                 row["raagi"].strip()) for row in entries)
                if any(end <= start or not name for start, end, name in ranges):
                    return None
                if any(previous[1] > following[0] for previous, following in zip(ranges, ranges[1:])):
                    return None
                minute = now.hour*60+now.minute
                return next((name for start, end, name in ranges if start <= minute < end), None)
            except (KeyError, TypeError, ValueError, AttributeError):
                return None
