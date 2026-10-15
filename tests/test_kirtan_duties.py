from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from scripts import update_kirtan_duties as duties


FIXTURES = Path(__file__).resolve().parents[1] / "validation/kirtan-duties/fixtures"
RAW = json.loads((FIXTURES/"sgpc-raw-pages.json").read_text(encoding="utf-8"))
PAGES = RAW["pages"]


def changed_pages():
    pages = deepcopy(PAGES)
    line = next(row for row in pages[0]["lines"] if row["columns"][0] == "02.10.2026")
    line["columns"][1] = "ਭਾਈ ਨਵੇਂ ਸਾਹਿਬ ਸਿੰਘ"
    line["text"] = "  ".join(line["columns"])
    return pages


def ocr_available():
    if not all(shutil.which(tool) for tool in ["pdftotext", "pdfinfo", "pdftoppm", "tesseract"]):
        return False
    result = subprocess.run(["tesseract", "--list-langs"], capture_output=True, timeout=15)
    return result.returncode == 0 and {b"pan", b"eng"}.issubset(set(result.stdout.splitlines()))


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
        self.assertEqual(set(data), {"duties", "updated_at"})
        self.assertEqual(data["duties"]["02-10-2026"][0]["raagi"], "ਭਾਈ ਗੁਰਮੇਲ ਸਿੰਘ")
        cache = json.loads(duties.metadata_path(self.output).read_text())
        self.assertEqual(cache["pages"], PAGES)
        self.assertEqual(cache["etag"], '"week-1"')
        self.assertEqual(cache["source_url"], duties.SOURCE_URL)
        self.assertIn("ਭਾਈ".encode(), self.output.read_bytes())

    def test_identical_pdf_keeps_bytes_and_modification_time(self):
        self.refresh();before = self.snapshot()
        with patch.object(duties, "extract_pages", side_effect=AssertionError("unnecessary extraction")):
            client = self.client(lambda request: httpx.Response(200, content=b"%PDF-fake"))
            self.assertFalse(duties.refresh(self.output, client))
        self.assertEqual(self.snapshot(), before)

    def test_regenerated_pdf_and_layout_spacing_are_not_new_data(self):
        self.refresh();before = self.snapshot()
        pages = deepcopy(PAGES)
        pages[0]["lines"].append({"text": "ਬਦਲਿਆ ਨੋਟ", "columns": ["ਬਦਲਿਆ ਨੋਟ"]})
        self.assertFalse(self.refresh(body=b"%PDF-regenerated", pages=pages))
        self.assertEqual(self.snapshot(), before)

    def test_changed_duties_replace_previous_data(self):
        self.refresh();before = self.output.read_bytes()
        pages = changed_pages()
        self.assertTrue(self.refresh(body=b"%PDF-new", pages=pages))
        self.assertNotEqual(self.output.read_bytes(), before)
        self.assertEqual(json.loads(self.output.read_text())["duties"]["02-10-2026"][0]["raagi"], "ਭਾਈ ਨਵੇਂ ਸਾਹਿਬ ਸਿੰਘ")

    def test_full_attachment_has_17_days_and_255_time_assignments(self):
        schedule = duties.parse_duties(PAGES)
        self.assertEqual(list(schedule), [f"{day:02d}-10-2026" for day in range(1,18)])
        self.assertTrue(all(len(entries) == 15 for entries in schedule.values()))
        self.assertEqual(sum(map(len, schedule.values())), 255)
        for entries in schedule.values():
            self.assertTrue(all(set(row) == {"start_time", "end_time", "raagi"} for row in entries))
            self.assertEqual(entries[-1]["start_time"], "20:45")
            self.assertEqual(entries[-1]["end_time"], "")

    def test_october_2_exact_morning_evening_and_midday_assignments(self):
        schedule = duties.parse_duties(PAGES)["02-10-2026"]
        self.assertEqual([(row["start_time"], row["end_time"]) for row in schedule],
                         [("02:45","03:45"),("03:45","06:45"),("06:50","08:00"),
                          ("08:00","09:00"),("09:00","10:00"),("10:00","11:00"),
                          ("11:00","12:00"),("12:00","13:10"),("13:10","14:20"),
                          ("14:20","15:20"),("15:20","16:20"),("16:20","17:35"),
                          ("17:35","18:50"),("19:10","20:45"),("20:45","")])
        self.assertEqual([row["raagi"] for row in schedule],
                         ["ਭਾਈ ਗੁਰਮੇਲ ਸਿੰਘ","ਭਾਈ ਸਤਿੰਦਰਬੀਰ ਸਿੰਘ","ਭਾਈ ਜਗਤਾਰ ਸਿੰਘ ਰਾ",
                          "ਭਾਈ ਆਗਿਆਕਾਰ ਸਿੰਘ","ਭਾਈ ਸੁਖਬੀਰ ਸਿੰਘ","ਭਾਈ ਭੁਪਿੰਦਰ ਸਿੰਘ",
                          "ਭਾਈ ਵਰਿੰਦਰ ਸਿੰਘ","ਭਾਈ ਕੁਲਬੀਰ ਸਿੰਘ","ਭਾਈ ਗੁਰਮੇਲ ਸਿੰਘ",
                          "ਭਾਈ ਜਗਤਾਰ ਸਿੰਘ ਰਾ","ਭਾਈ ਆਗਿਆਕਾਰ ਸਿੰਘ","ਭਾਈ ਸੁਖਬੀਰ ਸਿੰਘ",
                          "ਭਾਈ ਭੁਪਿੰਦਰ ਸਿੰਘ","ਭਾਈ ਵਰਿੰਦਰ ਸਿੰਘ","ਭਾਈ ਕੁਲਬੀਰ ਸਿੰਘ"])

    def test_updated_time_is_read_from_header_instead_of_hardcoded(self):
        pages = deepcopy(PAGES)
        pages[0]["lines"][4]["columns"][0] = "ਤਿੰਨ ਪਹਿਰਾ 2.40 ਤੋਂ"
        self.assertEqual(duties.parse_duties(pages)["02-10-2026"][0]["start_time"], "02:40")

    def test_incomplete_and_ambiguous_tables_preserve_existing_schedule(self):
        self.refresh();before = self.snapshot()
        incomplete = deepcopy(PAGES)
        del incomplete[0]["lines"][31]  # second table loses October 2
        missing_both = deepcopy(PAGES)
        missing_both[0]["lines"] = [line for line in missing_both[0]["lines"]
                                     if line["columns"][0] != "02.10.2026"]
        unknown = deepcopy(PAGES)
        unknown[0]["lines"][3]["columns"] = ["1","2","3","9"]
        overlap = deepcopy(PAGES)
        overlap[0]["lines"][5]["columns"][1] = "6.40 ਤੋਂ 8.00 ਸਵੇਰੇ"
        for pages in [incomplete, missing_both, unknown, overlap]:
            with self.subTest(pages=pages[0]["lines"][3]), self.assertRaises(ValueError):
                self.refresh(body=b"%PDF-new-invalid", pages=pages)
            self.assertEqual(self.snapshot(), before)

    def test_conflicting_date_row_is_rejected(self):
        pages = changed_pages()
        old_row = deepcopy(PAGES[0]["lines"][9])
        pages[0]["lines"].insert(10, old_row)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            duties.parse_duties(pages)

    def test_raw_json_migrates_even_if_pdf_has_identical_bytes(self):
        raw = {**RAW, "pdf_sha256": duties.hashlib.sha256(b"%PDF-fake").hexdigest()}
        self.output.write_text(json.dumps(raw, ensure_ascii=False))
        with patch.object(duties, "extract_pages", side_effect=AssertionError("should reuse old extracted pages")):
            self.assertTrue(duties.refresh(self.output, self.client(lambda r: httpx.Response(200, content=b"%PDF-fake"))))
        self.assertEqual(set(json.loads(self.output.read_text())), {"duties", "updated_at"})

    def test_raw_json_migrates_on_http_304(self):
        self.output.write_text(json.dumps(RAW, ensure_ascii=False))
        self.assertTrue(duties.refresh(self.output, self.client(lambda r: httpx.Response(304))))
        self.assertEqual(len(json.loads(self.output.read_text())["duties"]),17)

    def test_offline_same_file_migration_is_idempotent_and_uses_no_network(self):
        self.output.write_text(json.dumps(RAW, ensure_ascii=False))
        with patch.object(duties.httpx, "Client", side_effect=AssertionError("migration must not fetch")):
            self.assertEqual(duties.main(["--from-json",str(self.output),"--output",str(self.output)]),0)
            before = self.snapshot()
            self.assertEqual(duties.main(["--from-json",str(self.output),"--output",str(self.output)]),0)
        self.assertEqual(self.snapshot(),before)

    def test_output_write_failure_does_not_allow_a_stale_cache_to_skip_update(self):
        self.refresh(headers={"etag":'"old"'});before = self.snapshot()
        replace = duties.os.replace
        def fail_main(source,target):
            if target == self.output:
                raise OSError("failed after metadata commit")
            return replace(source,target)
        with patch.object(duties.os,"replace",side_effect=fail_main), self.assertRaises(OSError):
            self.refresh(body=b"%PDF-new",pages=changed_pages(),headers={"etag":'"new"'})
        self.assertEqual(self.snapshot(),before)
        def response(request):
            self.assertNotIn("if-none-match",request.headers)
            return httpx.Response(200,content=b"%PDF-new")
        with patch.object(duties,"extract_pages",return_value=changed_pages()):
            self.assertTrue(duties.refresh(self.output,self.client(response)))
        self.assertEqual(json.loads(self.output.read_text())["duties"]["02-10-2026"][0]["raagi"],"ਭਾਈ ਨਵੇਂ ਸਾਹਿਬ ਸਿੰਘ")

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
        self.assertEqual(json.loads(duties.metadata_path(self.output).read_text())["pdf_url"], "https://hs.sgpc.net/files/punjabi.pdf")

    def test_atomic_replace_failure_retains_previous_json(self):
        self.refresh();before = self.snapshot()
        with patch.object(duties.os, "replace", side_effect=OSError("disk error")), self.assertRaises(OSError):
            self.refresh(body=b"%PDF-new", pages=changed_pages())
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

    def test_unmapped_glyphs_are_not_accepted_as_unicode_punjabi(self):
        self.assertFalse(duties.readable_punjabi("ਭਾਈ\x03ਸਿੰਘ"))
        self.assertFalse(duties.readable_punjabi("ਭਾਈ\ue001ਸਿੰਘ"))
        self.assertTrue(duties.readable_punjabi("ਭਾਈ ਸਾਹਿਬ\n02:00\f"))

    def test_confirmed_sgpc_legacy_fonts_are_recognized(self):
        fixture = Path(__file__).resolve().parents[1] / "validation/kirtan-duties/fixtures/sgpc-pdffonts.txt"
        with patch.object(duties.subprocess, "run", return_value=SimpleNamespace(stdout=fixture.read_bytes())):
            self.assertEqual(duties.legacy_font_names(self.output),
                             ["GurbaniAkharThick", "AnmolLipiHeavy", "AnmolLipi,Bold", "AnmolLipi"])
        mixed = fixture.read_bytes()+b"Arial TrueType WinAnsi yes no no 20 0\n"
        with patch.object(duties.subprocess, "run", return_value=SimpleNamespace(stdout=mixed)):
            self.assertEqual(duties.legacy_font_names(self.output), [])

    def test_provided_sgpc_excerpt_converts_names_dates_and_columns(self):
        fixture = Path(__file__).resolve().parents[1] / "validation/kirtan-duties/fixtures/sgpc-extracted-excerpt.txt"
        text = duties.convert_legacy_text(fixture.read_text())
        self.assertIn("ਕੀਰਤਨ ਨਿਰਮੋਲਕ ਹੀਰਾ", text)
        self.assertIn("ਮਿਤੀ 01.10.2026 ਤੋਂ 17.10.2026", text)
        self.assertIn("ਆਸਾ ਦੀ ਵਾਰ 3.45", text)
        rows = [line for line in duties.page_lines(text) if line["text"].startswith("15.10.2026")]
        self.assertEqual(rows[0]["columns"], ["15.10.2026", "ਭਾਈ ਮਨਿੰਦਰ ਸਿੰਘ", "ਭਾਈ ਸ਼ੌਕੀਨ ਸਿੰਘ",
                                              "ਭਾਈ ਗੁਰਦੇਵ ਸਿੰਘ ਖ਼ਾ", "ਭਾਈ ਹਰਦੀਪ ਸਿੰਘ"])
        self.assertEqual(len([line for line in text.splitlines() if line.strip().startswith(tuple(f"{i:02d}.10.2026" for i in range(1,18)))]),17)

    @unittest.skipUnless(shutil.which("pdftotext") and shutil.which("pdffonts"), "Install Poppler")
    def test_legacy_pdf_direct_conversion_needs_no_ocr(self):
        pdf = unicode_pdf("15.10.2026     BweI minMdr isMG").replace(b"/Helvetica", b"/AnmolLipi")
        with patch.object(duties, "ocr_pages", side_effect=AssertionError("known font must not use OCR")):
            pages = duties.extract_pages(pdf)
            self.assertEqual(pages[0]["extraction_method"], "legacy-font-conversion")
            self.assertIn("15.10.2026", pages[0]["lines"][0]["text"])
            self.assertIn("ਭਾਈ ਮਨਿੰਦਰ ਸਿੰਘ", pages[0]["lines"][0]["text"])

    def test_missing_ocr_language_explains_installation(self):
        result = SimpleNamespace(stdout=b"List of available languages (1):\neng\n")
        with patch.object(duties.subprocess, "run", return_value=result), self.assertRaisesRegex(ValueError, "tesseract-ocr-pan"):
            duties.ocr_pages(self.output, self.output.parent)

    def run_mock_ocr(self, confidence=95, text="ਭਾਈ ਹਰਜਿੰਦਰ ਸਿੰਘ 02:00"):
        def command(args, **kwargs):
            if args[1] == "--list-langs":
                return SimpleNamespace(stdout=b"languages:\neng\npan\n")
            if args[0] == "pdfinfo":
                return SimpleNamespace(stdout=b"Pages: 1\n")
            if args[0] == "pdftoppm":
                Path(args[-1]+".png").write_bytes(b"image")
            else:
                Path(args[2]+".txt").write_text(text, encoding="utf-8")
                Path(args[2]+".tsv").write_text(
                    "level\tconf\ttext\n"+"".join(f"5\t{confidence}\t{word}\n" for word in text.split()), encoding="utf-8")
                self.assertEqual(kwargs["env"]["OMP_THREAD_LIMIT"], "1")
            return SimpleNamespace(stdout=b"")
        return patch.object(duties.subprocess, "run", side_effect=command)

    def test_ocr_rows_include_method_and_quality_metadata(self):
        with self.run_mock_ocr():
            pages = duties.ocr_pages(self.output, self.output.parent)
        self.assertEqual(pages[0]["extraction_method"], "ocr")
        self.assertEqual(pages[0]["ocr_mean_confidence"], 95)
        self.assertIn("ਹਰਜਿੰਦਰ", pages[0]["lines"][0]["text"])
        self.assertFalse((self.output.parent/"page-1.png").exists())

    def test_low_quality_ocr_and_non_punjabi_results_are_rejected(self):
        for score, text in [(40, "ਭਾਈ ਸਾਹਿਬ"), (99, "No Punjabi duties")]:
            with self.subTest(score=score, text=text), self.run_mock_ocr(score, text), self.assertRaisesRegex(ValueError, "confidence too low"):
                duties.ocr_pages(self.output, self.output.parent)

    def test_page_limit_prevents_unbounded_ocr_work(self):
        def command(args, **kwargs):
            if args[1] == "--list-langs":
                return SimpleNamespace(stdout=b"pan\neng\n")
            self.assertEqual(args[0], "pdfinfo")
            return SimpleNamespace(stdout=b"Pages: 999\n")
        with patch.object(duties.subprocess, "run", side_effect=command), self.assertRaisesRegex(ValueError, "31-page"):
            duties.ocr_pages(self.output, self.output.parent)

    @unittest.skipUnless(shutil.which("pdftotext"), "Install poppler-utils")
    def test_ocr_failure_preserves_previous_json(self):
        self.refresh();before = self.snapshot()
        client = self.client(lambda r: httpx.Response(200, content=unicode_pdf("unmapped legacy text")))
        with patch.object(duties, "ocr_pages", side_effect=ValueError("low OCR quality")), self.assertRaises(ValueError):
            duties.refresh(self.output, client)
        self.assertEqual(self.snapshot(), before)

    @unittest.skipUnless(shutil.which("pdftotext"), "Install poppler-utils to verify real PDF extraction")
    def test_actual_poppler_extracts_punjabi_pdf(self):
        text = "ਭਾਈ ਸਾਹਿਬ 02:00"
        with patch.object(duties, "ocr_pages", side_effect=AssertionError("Unicode PDF must not need OCR")):
            pages = duties.extract_pages(unicode_pdf(text))
        self.assertEqual(" ".join(line["text"] for page in pages for line in page["lines"]), text)
        with self.assertRaises(ValueError):
            duties.extract_pages(unicode_pdf("No Punjabi text"))

    @unittest.skipUnless(ocr_available(), "Install Punjabi/English Tesseract and Poppler for real OCR checks")
    def test_actual_ocr_recovers_non_unicode_and_scanned_duty_rows(self):
        fixtures = Path(__file__).resolve().parents[1] / "validation/kirtan-duties/fixtures"
        expected = ["ਭਾਈ ਹਰਜਿੰਦਰ ਸਿੰਘ", "ਭਾਈ ਕਮਲਜੀਤ ਸਿੰਘ", "ਭਾਈ ਜਸਪਿੰਦਰ ਸਿੰਘ",
                    "02:00 - 04:00", "04:00 - 06:00", "06:00 - 08:00"]
        for name in ["synthetic-unmapped-font.pdf", "synthetic-scanned.pdf"]:
            with self.subTest(fixture=name):
                pdf = (fixtures/name).read_bytes()
                pages = duties.extract_pages(pdf)
                text = "\n".join(line["text"] for page in pages for line in page["lines"])
                for phrase in expected:
                    self.assertIn(phrase, text)
                self.assertEqual(pages[0]["extraction_method"], "ocr")
                self.assertGreaterEqual(pages[0]["ocr_mean_confidence"], duties.MIN_OCR_CONFIDENCE)
                with self.assertRaises(ValueError):
                    duties.parse_duties(pages)  # A generic test table is not the SGPC duty layout.


if __name__ == "__main__":
    unittest.main()
