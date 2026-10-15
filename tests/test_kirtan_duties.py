import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import httpx

from scripts import update_kirtan_duties as duties


PAGES = [{"page": 1, "lines": [{"text": "ਭਾਈ ਸਾਹਿਬ    02:00 - 04:00",
                                    "columns": ["ਭਾਈ ਸਾਹਿਬ", "02:00 - 04:00"]}]}]


def unicode_pdf(text):
    """A small real PDF with Punjabi Unicode mappings, without test dependencies."""
    chars = list(dict.fromkeys(text))
    codes = {char: index + 32 for index, char in enumerate(chars)}
    mappings = "\n".join(f"<{codes[c]:02x}> <{c.encode('utf-16-be').hex()}>" for c in chars)
    cmap = ("/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
            "/CMapName /Test def /CMapType 2 def\n"
            "1 begincodespacerange <00> <ff> endcodespacerange\n"
            f"{len(chars)} beginbfchar\n{mappings}\nendbfchar\n"
            "endcmap CMapName currentdict /CMap defineresource pop end end").encode()
    content = ("BT /F1 12 Tf 20 700 Td <" + "".join(f"{codes[c]:02x}" for c in text) + "> Tj ET").encode()
    def stream(data):
        return f"<< /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>",
               stream(content), stream(cmap)]
    pdf = b"%PDF-1.4\n";offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf));pdf += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 7\n0000000000 65535 f \n"
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    return pdf + f"trailer << /Size 7 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()


class DutiesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "duties.json"

    def client(self, handler):
        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        self.addCleanup(client.close)
        return client

    def refresh(self, body=b"%PDF-fake", pages=PAGES, headers=None):
        client = self.client(lambda request: httpx.Response(200, content=body, headers=headers or {}))
        with patch.object(duties, "extract_pages", return_value=pages):
            return duties.refresh(self.output, client)

    def snapshot(self):
        return self.output.read_bytes(), self.output.stat().st_mtime_ns

    def test_first_refresh_writes_unicode_json_and_metadata(self):
        self.assertTrue(self.refresh(headers={"etag": '"week-1"'}))
        data = json.loads(self.output.read_text())
        self.assertEqual(data["pages"], PAGES)
        self.assertEqual(data["etag"], '"week-1"')
        self.assertEqual(data["source_url"], duties.SOURCE_URL)
        self.assertIn("ਭਾਈ".encode(), self.output.read_bytes())

    def test_identical_pdf_keeps_bytes_and_modification_time(self):
        self.refresh();before = self.snapshot()
        with patch.object(duties, "extract_pages", side_effect=AssertionError("unnecessary extraction")):
            client = self.client(lambda request: httpx.Response(200, content=b"%PDF-fake"))
            self.assertFalse(duties.refresh(self.output, client))
        self.assertEqual(self.snapshot(), before)

    def test_regenerated_pdf_and_layout_spacing_are_not_new_data(self):
        self.refresh();before = self.snapshot()
        pages = [{"page": 1, "lines": [{"text": "ਭਾਈ ਸਾਹਿਬ  02:00 - 04:00", "columns": []}]}]
        self.assertFalse(self.refresh(body=b"%PDF-regenerated", pages=pages))
        self.assertEqual(self.snapshot(), before)

    def test_changed_duties_replace_previous_data(self):
        self.refresh();before = self.output.read_bytes()
        pages = [{"page": 1, "lines": [{"text": "ਭਾਈ ਦੂਜੇ ਸਾਹਿਬ  04:00 - 06:00", "columns": []}]}]
        self.assertTrue(self.refresh(body=b"%PDF-new", pages=pages))
        self.assertNotEqual(self.output.read_bytes(), before)
        self.assertEqual(json.loads(self.output.read_text())["pages"], pages)

    def test_304_sends_validators_and_keeps_file_untouched(self):
        self.refresh(headers={"etag": '"same"', "last-modified": "Wed, 14 Oct 2026 00:00:00 GMT"})
        before = self.snapshot()
        def response(request):
            self.assertEqual(request.headers["if-none-match"], '"same"')
            self.assertIn("if-modified-since", request.headers)
            return httpx.Response(304)
        self.assertFalse(duties.refresh(self.output, self.client(response)))
        self.assertEqual(self.snapshot(), before)

    def test_http_timeout_invalid_page_and_pdf_failures_retain_file(self):
        self.refresh();before = self.snapshot()
        def timeout(request):
            raise httpx.ReadTimeout("timeout", request=request)
        handlers = [lambda r: httpx.Response(503), timeout,
                    lambda r: httpx.Response(200, text="<html>Temporarily unavailable</html>")]
        for handler in handlers:
            with self.subTest(handler=handler), self.assertRaises((httpx.HTTPError, ValueError)):
                duties.refresh(self.output, self.client(handler))
            self.assertEqual(self.snapshot(), before)
        with patch.object(duties, "extract_pages", side_effect=ValueError("unreadable PDF")), self.assertRaises(ValueError):
            duties.refresh(self.output, self.client(lambda r: httpx.Response(200, content=b"%PDF-bad")))
        self.assertEqual(self.snapshot(), before)

    def test_embedded_pdf_resolves_relative_url(self):
        def response(request):
            if request.url.path.endswith(".php"):
                return httpx.Response(200, text='<iframe src="files/punjabi.pdf"></iframe>')
            self.assertEqual(str(request.url), "https://hs.sgpc.net/files/punjabi.pdf")
            return httpx.Response(200, content=b"%PDF-test")
        with patch.object(duties, "extract_pages", return_value=PAGES):
            self.assertTrue(duties.refresh(self.output, self.client(response)))
        self.assertEqual(json.loads(self.output.read_text())["pdf_url"], "https://hs.sgpc.net/files/punjabi.pdf")

    def test_atomic_replace_failure_retains_previous_json(self):
        self.refresh();before = self.snapshot()
        with patch.object(duties.os, "replace", side_effect=OSError("disk error")), self.assertRaises(OSError):
            self.refresh(body=b"%PDF-new", pages=[{"lines": [{"text": "ਨਵੀਂ ਡਿਊਟੀ"}]}])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(list(self.output.parent.glob(".duties.json.*")), [])

    def test_concurrent_run_cannot_fetch_or_write(self):
        with duties.output_lock(self.output), self.assertRaises(BlockingIOError):
            duties.refresh(self.output, self.client(lambda r: self.fail("concurrent fetch")))

    def test_failed_first_download_does_not_create_json(self):
        with self.assertRaises(httpx.HTTPError):
            duties.refresh(self.output, self.client(lambda r: httpx.Response(503)))
        self.assertFalse(self.output.exists())

    def test_oversized_download_preserves_existing_json(self):
        self.refresh();before = self.snapshot()
        with patch.object(duties, "MAX_DOWNLOAD_BYTES", 10), self.assertRaises(ValueError):
            self.refresh(body=b"%PDF-" + b"x" * 20)
        self.assertEqual(self.snapshot(), before)

    def test_cli_failure_returns_nonzero_and_keeps_previous_json(self):
        self.refresh();before = self.snapshot()
        client = self.client(lambda r: httpx.Response(503))
        with patch.object(duties.httpx, "Client", return_value=client), self.assertLogs(duties.LOGGER, level="ERROR"):
            self.assertEqual(duties.main(["--output", str(self.output)]), 1)
        self.assertEqual(self.snapshot(), before)

    @unittest.skipUnless(shutil.which("pdftotext"), "Install poppler-utils to verify real PDF extraction")
    def test_actual_poppler_extracts_punjabi_pdf(self):
        text = "ਭਾਈ ਸਾਹਿਬ 02:00"
        pages = duties.extract_pages(unicode_pdf(text))
        self.assertEqual(" ".join(line["text"] for page in pages for line in page["lines"]), text)
        with self.assertRaises(ValueError):
            duties.extract_pages(unicode_pdf("No Punjabi text"))


if __name__ == "__main__":
    unittest.main()
