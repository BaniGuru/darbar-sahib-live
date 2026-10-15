"""Refresh the SGPC Punjabi duties JSON without replacing good data on failure."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
from html.parser import HTMLParser
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata
from urllib.parse import urljoin, urlsplit

import httpx
from gurmukhiutils.unicode import unicode as gurmukhi_unicode


SOURCE_URL = "https://hs.sgpc.net/kirtan_duty_punjabi_pdf.php"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/kirtan_duties.json"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_OCR_PAGES = 31
MIN_OCR_CONFIDENCE = 60
LOGGER = logging.getLogger("kirtan-duties")


class PDFLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        target = attrs.get("data" if tag == "object" else "href" if tag == "a" else "src")
        if target and (tag in {"iframe", "embed", "object"}
                       or tag == "a" and urlsplit(target).path.lower().endswith(".pdf")):
            self.links.append(target)


def download_pdf(client, existing):
    """Accept a direct PDF or a page embedding one PDF on the SGPC host."""
    url = SOURCE_URL
    visited = set()
    for _ in range(4):
        if url in visited:
            raise ValueError("Duties page links back to itself")
        visited.add(url)
        headers = {}
        if existing and url == existing.get("pdf_url"):
            if existing.get("etag"):
                headers["If-None-Match"] = existing["etag"]
            if existing.get("last_modified"):
                headers["If-Modified-Since"] = existing["last_modified"]
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 304:
                if not existing:
                    raise ValueError("Received not-modified without existing duties")
                return None
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Duties download exceeds 20 MiB")
            pdf_url = str(response.url)
            if b"%PDF-" in body[:1024]:
                return bytes(body), pdf_url, dict(response.headers)
            parser = PDFLinks()
            parser.feed(bytes(body).decode("utf-8", errors="replace"))
            candidates = list(dict.fromkeys(urljoin(pdf_url, link) for link in parser.links))
            candidates = [link for link in candidates if urlsplit(link).scheme == "https"
                          and urlsplit(link).hostname == urlsplit(SOURCE_URL).hostname]
            if len(candidates) != 1:
                raise ValueError("Expected a PDF or one embedded SGPC duties PDF")
            url = candidates[0]
    raise ValueError("Too many nested duties pages")


def page_lines(text):
    return [{"text": line.strip(), "columns": re.split(r"\s{2,}", line.strip())}
            for line in text.splitlines() if line.strip()]


def readable_punjabi(text):
    unmapped = any(unicodedata.category(char) in {"Co", "Cs"}
                   or unicodedata.category(char) == "Cc" and char not in "\t\n\r\f"
                   for char in text)
    return bool(re.search(r"[\u0a00-\u0a7f]", text)) and "\ufffd" not in text and not unmapped


def legacy_font_names(path):
    """Whole-page conversion is safe only if every font uses this mapping."""
    result = subprocess.run(["pdffonts", str(path)], capture_output=True, check=True, timeout=15)
    names = [re.sub(r"^[A-Z]{6}\+", "", line.split()[0])
             for line in result.stdout.decode("utf-8").splitlines()[2:] if line.strip()]
    if names and all(re.match(r"^(AnmolLipi|(?:Open)?GurbaniAkhar)", name, re.I) for name in names):
        return names
    return []


def convert_legacy_text(text):
    # Keep dates/times/slot numbers machine-readable and preserve their values.
    return unicodedata.normalize("NFC", gurmukhi_unicode(text)).translate(
        str.maketrans("੦੧੨੩੪੫੬੭੮੯", "0123456789"))


def ocr_pages(path: Path, directory: Path):
    """Render unextractable fonts/scans; never transliterate guessed ASCII."""
    try:
        installed = subprocess.run(["tesseract", "--list-langs"], capture_output=True,
                                   check=True, timeout=15).stdout.decode().splitlines()
    except FileNotFoundError as error:
        raise ValueError("Punjabi PDF needs OCR. Install: sudo apt-get install poppler-utils "
                         "tesseract-ocr tesseract-ocr-pan tesseract-ocr-eng") from error
    if not {"pan", "eng"}.issubset(set(installed)):
        raise ValueError("Punjabi PDF needs OCR language data. Install: sudo apt-get install "
                         "tesseract-ocr-pan tesseract-ocr-eng")
    info = subprocess.run(["pdfinfo", str(path)], capture_output=True, check=True, timeout=15)
    count = re.search(rb"^Pages:\s+(\d+)", info.stdout, re.MULTILINE)
    if count is None or not 1 <= int(count[1]) <= MAX_OCR_PAGES:
        raise ValueError("PDF page count is unknown or exceeds the 31-page OCR limit")
    pages = []
    # Keep this nightly task bounded on the same small CPU used by the live ASR.
    environment = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    for number in range(1, int(count[1])+1):
        image = directory / f"page-{number}"
        result = directory / f"ocr-{number}"
        subprocess.run(["pdftoppm", "-f", str(number), "-l", str(number), "-r", "300",
                        "-gray", "-png", "-singlefile", str(path), str(image)],
                       capture_output=True, check=True, timeout=90, env=environment)
        subprocess.run(["tesseract", str(image)+".png", str(result), "-l", "pan+eng",
                        "--psm", "6", "-c", "preserve_interword_spaces=1", "txt", "tsv"],
                       capture_output=True, check=True, timeout=120, env=environment)
        text = unicodedata.normalize("NFC", result.with_suffix(".txt").read_text(encoding="utf-8"))
        with result.with_suffix(".tsv").open(encoding="utf-8") as stream:
            words = [row for row in csv.DictReader(stream, delimiter="\t")
                     if row.get("level") == "5" and row.get("text", "").strip()]
        punjabi = [word for word in words if readable_punjabi(word["text"])]
        def confidence(records):
            weight = sum(len(word["text"]) for word in records)
            return sum(max(0, float(word["conf"]))*len(word["text"]) for word in records)/weight if weight else 0
        score = min(confidence(words), confidence(punjabi))
        if not readable_punjabi(text) or not punjabi or score < MIN_OCR_CONFIDENCE:
            raise ValueError(f"Punjabi OCR confidence too low on page {number} ({score:.1f}); keeping previous duties")
        pages.append({"page": number, "lines": page_lines(text), "extraction_method": "ocr",
                      "ocr_mean_confidence": round(score, 2)})
        LOGGER.warning("Duties page %d used Punjabi OCR (confidence %.1f); check names/times against PDF", number, score)
        # Retain only one page image at a time.
        image.with_suffix(".png").unlink()
    return pages


def extract_pages(pdf: bytes):
    """Prefer Unicode or confirmed legacy font maps; OCR other unreadable PDFs."""
    with tempfile.TemporaryDirectory(prefix="kirtan-duties-") as directory:
        path = Path(directory) / "duties.pdf"
        path.write_bytes(pdf)
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", "-eol", "unix", str(path), "-"],
            capture_output=True, check=True, timeout=60,
        )
        text = unicodedata.normalize("NFC", result.stdout.decode("utf-8"))
        if not readable_punjabi(text):
            fonts = legacy_font_names(path)
            converted = convert_legacy_text(text) if fonts else ""
            if fonts and readable_punjabi(converted):
                LOGGER.info("Converting AnmolLipi/GurbaniAkhar font text directly to Unicode")
                return [{"page": number, "lines": page_lines(convert_legacy_text(page)),
                         "extraction_method": "legacy-font-conversion", "source_fonts": fonts}
                        for number, page in enumerate(text.split("\f"), 1) if page.strip()]
            LOGGER.info("PDF has no usable Unicode or supported legacy mapping; trying Punjabi OCR")
            return ocr_pages(path, Path(directory))
    pages = []
    for number, page in enumerate(text.split("\f"), 1):
        lines = page_lines(page)
        if lines:
            pages.append({"page": number, "lines": lines})
    if not pages:
        raise ValueError("PDF contains no duties text")
    return pages


def content_hash(pages):
    # Insignificant PDF layout spacing and regenerated metadata are not new data.
    lines = [[" ".join(line["text"].split()) for line in page["lines"]] for page in pages]
    return hashlib.sha256(json.dumps(lines, ensure_ascii=False).encode()).hexdigest()


DATE_PATTERN = r"(\d{2})[./-](\d{2})[./-](\d{4})"
TIME_PATTERN = r"(?<!\d)(\d{1,2})[.:](\d{2})(?![\d.])"


def _date(value):
    match = re.fullmatch(DATE_PATTERN, value)
    if not match:
        raise ValueError(f"Invalid duty date: {value}")
    return datetime(*(int(match[i]) for i in (3, 2, 1))).date()


def _time_range(cell, period):
    times = re.findall(TIME_PATTERN, cell)
    open_end = "ਸਮਾਪਤੀ" in cell
    if "ਤੋਂ" not in cell or len(times) != (1 if open_end else 2):
        raise ValueError(f"Unrecognized duty time range: {cell}")
    if "ਸ਼ਾਮ" in cell or "ਦੁਪਹਿਰ" in cell:
        period = "pm"
    values = []
    for raw_hour, raw_minute in times:
        hour, minute = int(raw_hour), int(raw_minute)
        if not 0 <= hour <= 23 or not 0 <= minute < 60:
            raise ValueError(f"Invalid duty time: {cell}")
        if period == "pm" and hour < 12:
            hour += 12
        values.append(f"{hour:02d}:{minute:02d}")
    if not open_end and values[0] >= values[1]:
        raise ValueError(f"Duty ends before it starts: {cell}")
    return values[0], "" if open_end else values[1]


def _table_times(ids, headers):
    """Reassemble the wrapped time headings in the numbered SGPC tables.

    Numeric values come from the PDF; a changed/unrecognized layout fails
    rather than silently applying a hardcoded daily timetable.
    """
    def find(predicate):
        matches = [row for row in headers if predicate(row)]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely read time headings for columns {ids}")
        return matches[0]

    def count(cell):
        return len(re.findall(TIME_PATTERN, cell))

    if ids == (1, 2, 3, 4):
        starts = find(lambda row: len(row) == 2 and all(count(cell) == 1 and "ਤੋਂ" in cell for cell in row))
        ends = find(lambda row: len(row) == 2 and all(count(cell) == 1 and "ਤੋਂ" not in cell for cell in row))
        right = find(lambda row: len(row) == 3 and row[0] == "ਮਿਤੀ" and all(count(cell) == 2 for cell in row[1:]))
        later = find(lambda row: len(row) == 4 and count(row[0]) == 2 and count(row[1]) == 0
                     and all(count(cell) == 2 for cell in row[2:]))
        result = {1: [_time_range(starts[0]+" "+ends[0], "am")],
                  2: [_time_range(starts[1]+" "+ends[1], "am")],
                  3: [_time_range(right[1], "am")], 4: [_time_range(right[2], "am")]}
        for slot, cell in zip((1, 3, 4), (later[0], later[2], later[3])):
            result[slot].append(_time_range(cell, "pm"))
        # "Broadcasting" labels Asa Ki Vaar, and supplies no extra time range.
        return result
    if ids == (5, 6, 7, 8):
        morning = find(lambda row: len(row) == 3 and all(count(cell) == 2 for cell in row))
        midday = find(lambda row: len(row) == 1 and count(row[0]) == 2)
        later = find(lambda row: len(row) == 4 and row[0] == "ਮਿਤੀ" and all(count(cell) == 2 for cell in row[1:]))
        last = find(lambda row: len(row) == 4 and all(count(cell) == 0 for cell in row[:3])
                    and "ਸਮਾਪਤੀ" in row[3])
        result = {slot: [_time_range(early, "am"), _time_range(late, "pm")]
                  for slot, early, late in zip((5, 6, 7), morning, later[1:])}
        result[8] = [_time_range(midday[0], "pm"), _time_range(last[3], "pm")]
        return result
    raise ValueError(f"Unsupported SGPC duty columns: {ids}")


def parse_duties(pages):
    """Build complete dated assignments, preserving gaps and unspecified ends."""
    if not isinstance(pages, list) or not pages:
        raise ValueError("No extracted duties pages")
    sections = []
    current = None
    declared_ranges = set()
    digit_map = str.maketrans("੦੧੨੩੪੫੬੭੮੯", "0123456789")
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("lines"), list):
            raise ValueError("Invalid extracted duties page")
        for line in page["lines"]:
            if not isinstance(line, dict):
                raise ValueError("Invalid extracted duties row")
            raw_columns = line.get("columns") or re.split(r"\s{2,}", line.get("text", "").strip())
            columns = [unicodedata.normalize("NFC", str(cell)).translate(digit_map).strip()
                       for cell in raw_columns]
            if not any(columns):
                continue
            text = " ".join(columns)
            if "ਮਿਤੀ" in text and "ਤੀਕ" in text:
                dates = re.findall(DATE_PATTERN, text)
                if len(dates) == 2:
                    declared_ranges.add(tuple(_date(".".join(date)) for date in dates))
            if len(columns) > 1 and all(cell.isdecimal() for cell in columns):
                ids = tuple(map(int, columns))
                if ids not in {(1, 2, 3, 4), (5, 6, 7, 8)}:
                    raise ValueError(f"Unsupported SGPC duty columns: {ids}")
                current = {"ids": ids, "headers": [], "rows": {}}
                sections.append(current)
            elif re.match(DATE_PATTERN+r"(?:\s|$)", columns[0]):
                if current is None or len(columns) != 5:
                    raise ValueError(f"Duty row lost its numbered table or four raagis: {text}")
                date = _date(columns[0])
                names = columns[1:]
                if not all(readable_punjabi(name) for name in names):
                    raise ValueError(f"Unreadable raagi on {date}")
                previous = current["rows"].get(date)
                if previous is not None and previous != names:
                    raise ValueError(f"Conflicting duty row on {date}")
                current["rows"][date] = names
            elif current is not None and not current["rows"]:
                current["headers"].append(columns)
    assignments = {}
    for section in sections:
        times = _table_times(section["ids"], section["headers"])
        for date, names in section["rows"].items():
            daily = assignments.setdefault(date, {})
            for slot, name in zip(section["ids"], names):
                assignment = (name, times[slot])
                if slot in daily and daily[slot] != assignment:
                    raise ValueError(f"Conflicting assignment on {date}, column {slot}")
                daily[slot] = assignment
    if not assignments:
        raise ValueError("No dated SGPC duty assignments found")
    if declared_ranges:
        if len(declared_ranges) != 1:
            raise ValueError("Conflicting published duty date ranges")
        first, last = next(iter(declared_ranges))
        if not 0 <= (last-first).days <= 366:
            raise ValueError("Invalid published duty date range")
        expected = {first+timedelta(days=day) for day in range((last-first).days+1)}
        if set(assignments) != expected:
            raise ValueError("Duties do not cover the complete published date range")
    result = {}
    for date, daily in sorted(assignments.items()):
        if set(daily) != set(range(1, 9)):
            raise ValueError(f"Incomplete raagi table on {date}")
        entries = sorted((dict(start_time=start, end_time=end, raagi=name)
                          for name, ranges in daily.values() for start, end in ranges),
                         key=lambda entry: entry["start_time"])
        for previous, following in zip(entries, entries[1:]):
            if not previous["end_time"] or previous["end_time"] > following["start_time"]:
                raise ValueError(f"Overlapping duty times on {date}")
        result[date.strftime("%d-%m-%Y")] = entries
    return result


def atomic_write(output: Path, document):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def metadata_path(output):
    return output.with_suffix(".meta.json")


def duties_hash(duties):
    return hashlib.sha256(json.dumps(duties, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_existing(output):
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
    if existing is not None and not isinstance(existing, dict):
        raise ValueError("Existing duties JSON is not an object")
    return existing


def _cache_for(output, existing):
    if existing and "pages" in existing:
        return existing  # Old output is its own extraction cache during migration.
    path = metadata_path(output)
    if not existing or not path.exists():
        return None
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        LOGGER.warning("Ignoring unreadable duties metadata; fetching without validators")
        return None
    # A crash between the two atomic writes must not hide an unapplied update.
    if isinstance(cache, dict) and cache.get("duties_sha256") == duties_hash(existing.get("duties")):
        return cache
    return None


def _store_schedule(output, pages, metadata, existing):
    duties = parse_duties(pages)
    changed = (not existing or set(existing) != {"duties", "updated_at"}
               or existing.get("duties") != duties)
    cache = {**metadata, "schema_version": 2, "pages": pages,
             "content_sha256": content_hash(pages), "duties_sha256": duties_hash(duties)}
    # Write provenance first. Its hash is checked against the main JSON on read.
    atomic_write(metadata_path(output), cache)
    if changed:
        atomic_write(output, {"duties": duties, "updated_at": datetime.now(timezone.utc).isoformat()})
    return changed


def convert_existing_json(source: Path, output: Path):
    """Migrate a saved extraction without requiring another network request."""
    with output_lock(output):
        data = json.loads(source.read_text(encoding="utf-8"))
        if source.resolve() == output.resolve() and isinstance(data, dict) and set(data) == {"duties", "updated_at"}:
            return False
        if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
            raise ValueError("--from-json requires the old JSON containing extracted pages")
        return _store_schedule(output, data["pages"], data, read_existing(output))


def refresh(output: Path, client):
    with output_lock(output):
        existing = read_existing(output)
        cache = _cache_for(output, existing)
        fetched = download_pdf(client, cache)
        if fetched is None:
            if existing and set(existing) == {"duties", "updated_at"}:
                return False
            return _store_schedule(output, cache["pages"], cache, existing)
        pdf, pdf_url, headers = fetched
        pdf_hash = hashlib.sha256(pdf).hexdigest()
        if cache and cache.get("pdf_sha256") == pdf_hash:
            if existing and set(existing) == {"duties", "updated_at"}:
                return False
            return _store_schedule(output, cache["pages"], cache, existing)
        pages = extract_pages(pdf)
        metadata = dict(source_url=SOURCE_URL, pdf_url=pdf_url,
                        fetched_at=datetime.now(timezone.utc).isoformat(), pdf_sha256=pdf_hash,
                        etag=headers.get("etag"), last_modified=headers.get("last-modified"))
        return _store_schedule(output, pages, metadata, existing)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--from-json", type=Path, help="Convert an existing raw-pages JSON without fetching the PDF")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.from_json:
            changed = convert_existing_json(args.from_json, args.output.resolve())
        else:
            with httpx.Client(timeout=30, follow_redirects=True, max_redirects=3,
                              headers={"User-Agent": "BaniGuru-kirtan-duties/1.0"}) as client:
                changed = refresh(args.output.resolve(), client)
        LOGGER.info("%s: %s", "Updated duties" if changed else "No new duties; file unchanged", args.output)
        return 0
    except BlockingIOError:
        LOGGER.info("Another duties refresh is running; skipped")
        return 0
    except (OSError, ValueError, httpx.HTTPError, subprocess.SubprocessError) as error:
        LOGGER.error("Refresh failed; previous JSON retained: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
