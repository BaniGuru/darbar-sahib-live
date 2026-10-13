"""Frame-aligned compressed audio buffering; no audio transcoding."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFrame:
    start: float
    end: float
    data: bytes


class FrameParser:
    """Incrementally extract ADTS AAC and MPEG audio frames, excluding ID3."""
    rates = (96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050,
             16000, 12000, 11025, 8000, 7350)

    def __init__(self, start=0.0):
        self.position = start
        self.buffer = bytearray()
        self.skip = 0

    def feed(self, data):
        self.buffer.extend(data)
        result = []
        while True:
            if self.skip:
                size = min(self.skip, len(self.buffer))
                del self.buffer[:size]
                self.skip -= size
                if self.skip:
                    break
            if len(self.buffer) < 10:
                break
            b = self.buffer
            if b[:3] == b'ID3':
                self.skip = 10 + sum((b[6+i] & 127) << (21-7*i) for i in range(4))
                if b[5] & 16:
                    self.skip += 10
                continue
            duration = None
            size = 0
            if b[0] == 255 and b[1] & 246 == 240:
                index = (b[2] >> 2) & 15
                size = ((b[3] & 3) << 11) | (b[4] << 3) | (b[5] >> 5)
                if index < len(self.rates) and size >= (7 if b[1] & 1 else 9):
                    duration = 1024 * ((b[6] & 3) + 1) / self.rates[index]
            elif b[0] == 255 and b[1] & 224 == 224:
                version, layer = (b[1] >> 3) & 3, (b[1] >> 1) & 3
                bi, ri = b[2] >> 4, (b[2] >> 2) & 3
                if version != 1 and layer and 0 < bi < 15 and ri < 3:
                    rate = (44100, 48000, 32000)[ri] // (1 if version == 3 else 2 if version == 2 else 4)
                    tables = {3: (32,64,96,128,160,192,224,256,288,320,352,384,416,448),
                              2: (32,48,56,64,80,96,112,128,160,192,224,256,320,384),
                              1: (32,40,48,56,64,80,96,112,128,160,192,224,256,320)}
                    low = (32,48,56,64,80,96,112,128,144,160,176,192,224,256) if layer == 3 else (8,16,24,32,40,48,56,64,80,96,112,128,144,160)
                    bitrate = (tables[layer] if version == 3 else low)[bi-1] * 1000
                    padding = (b[2] >> 1) & 1
                    samples = 384 if layer == 3 else 576 if layer == 1 and version != 3 else 1152
                    size = ((12 * bitrate // rate) + padding) * 4 if layer == 3 else (samples // 8) * bitrate // rate + padding
                    duration = samples / rate
            if duration is None:
                del b[0]
                continue
            if len(b) < size:
                break
            start = self.position
            self.position += duration
            result.append(AudioFrame(start, self.position, bytes(b[:size])))
            del b[:size]
        return result


class DelayedAudio:
    """Bounded media-clock delay; wall age drains buffered audio on outage."""
    def __init__(self, delay=10.0):
        self.delay = delay
        self.head = 0.0
        self.pending = deque()
        self.released = deque()
        self.epochs = {}
        self.segment = 0

    def new_segment(self):
        self.segment += 1

    def append(self, frames, now):
        if not frames:
            return
        self.head = frames[-1].end
        self.epochs[self.segment] = min(self.epochs.get(self.segment, now-self.head), now-self.head)
        self.pending.extend((self.segment, f) for f in frames)

    def release(self, now):
        result = []
        while self.pending:
            segment, frame = self.pending[0]
            if frame.end > self.head-self.delay and self.epochs[segment]+frame.end > now-self.delay:
                break
            self.pending.popleft()
            self.released.append(frame)
            result.append(frame)
        while self.released and self.released[0].end < self.head-self.delay-10:
            self.released.popleft()
        retained = {s for s, _ in self.pending} | {self.segment}
        self.epochs = {s: t for s, t in self.epochs.items() if s in retained}
        return result

    def bootstrap(self):
        """Start near the delayed edge, with MP3 reservoir preroll."""
        if not self.released:
            return []
        edge = self.released[-1].end
        return [f for f in self.released if f.end > edge-0.25]
