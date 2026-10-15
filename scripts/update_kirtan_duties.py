"""Refresh the SGPC Punjabi duties JSON without replacing good data on failure."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
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


def refresh(output: Path, client):
    with output_lock(output):
        existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
        if existing is not None and not isinstance(existing, dict):
            raise ValueError("Existing duties JSON is not an object")
        fetched = download_pdf(client, existing)
        if fetched is None:
            return False
        pdf, pdf_url, headers = fetched
        pdf_hash = hashlib.sha256(pdf).hexdigest()
        if existing and existing.get("pdf_sha256") == pdf_hash:
            return False
        pages = extract_pages(pdf)
        digest = content_hash(pages)
        if existing and existing.get("content_sha256") == digest:
            return False
        document = dict(
            schema_version=1, source_url=SOURCE_URL, pdf_url=pdf_url,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            content_sha256=digest, pdf_sha256=pdf_hash,
            etag=headers.get("etag"), last_modified=headers.get("last-modified"), pages=pages,
        )
        atomic_write(output, document)
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
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
