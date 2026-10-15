The nightly job fetches the Punjabi kirtan duties from
<https://hs.sgpc.net/kirtan_duty_punjabi_pdf.php> and saves
`data/kirtan_duties.json`. It runs separately from the live ASR service.

The output has exactly two fields: `duties` and `updated_at`. Dates are
`DD-MM-YYYY`; each date contains its assignments in time order. Times are
24-hour `HH:MM` in **Asia/Kolkata**, and raagi names remain Punjabi.
An abbreviated example from the supplied publication:

```json
{
  "duties": {
    "02-10-2026": [
      {"start_time": "02:45", "end_time": "03:45", "raagi": "ਭਾਈ ਗੁਰਮੇਲ ਸਿੰਘ"},
      {"start_time": "03:45", "end_time": "06:45", "raagi": "ਭਾਈ ਸਤਿੰਦਰਬੀਰ ਸਿੰਘ"}
    ]
  },
  "updated_at": "2026-10-15T05:39:23+00:00"
}
```

`updated_at` is the UTC ISO timestamp when this JSON changes, including
initial format migration. The supplied table has 15 performances per date:
eight morning/midday slots and seven afternoon/evening slots. The same
raagi is assigned to both timed performances in their PDF column. Asa Ki
Vaar has one timed performance; “broadcasting” alone adds no invented time.
The final `20:45` performance ends at “ਸਮਾਪਤੀ”, so its `end_time` is `""`.
Five- and twenty-minute gaps in the published schedule are preserved.

Times are read from the numbered table headings, including wrapped ranges,
not from a fixed timetable. Missing/conflicting raagis, incomplete published
date coverage, overlapping times or unrecognized layouts fail safely.
Source URLs, extracted pages, PDF/content hashes, HTTP validators and a
hash of the structured duties move to ignored `data/kirtan_duties.meta.json`.
The metadata hash prevents a failed main-file write from making a later
run incorrectly skip an unapplied update.

New duties replace the JSON atomically. Unchanged PDF bytes, HTTP 304, or
regenerated PDFs with the same assignments leave its **contents and
modification time unchanged**. Timeouts, HTTP errors, invalid HTML/PDFs,
missing extraction tools and unreadable/low-confidence Punjabi PDFs
retain the last good JSON and exit nonzero. Concurrent runs are locked.
A failed first fetch creates no placeholder duties JSON. Generated duties,
lock, metadata and log files are ignored by Git. Metadata may refresh
without changing the public JSON or `updated_at`.

On the production server, install Poppler and cron if absent:

```bash
sudo apt-get install poppler-utils cron
cd /var/www/gurbani-asr
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/update_kirtan_duties.py
```

The script uses existing `httpx` and pinned **gurmukhiutils 0.3.1**, a small
pure-Python converter for the confirmed SGPC fonts. The application
owner must be able to write `data/`. Its default output is resolved from
the script location, so cron does not need a working directory. An
alternative location can be supplied with `--output /path/to/duties.json`.

Existing raw-pages JSON is migrated on the next successful refresh, even
if the PDF is byte-identical or the server returns HTTP 304. To convert
the already downloaded JSON immediately **without network access**:

```bash
cd /var/www/gurbani-asr
.venv/bin/python scripts/update_kirtan_duties.py --from-json data/kirtan_duties.json
```

The old file is read completely before atomically replacing it; repeating
this in-place command leaves the new format unchanged. To retain the input,
add `--output /path/to/new-duties.json`. The nightly cron command is unchanged.

The user's `pdffonts` output confirms `GurbaniAkharThick`, `AnmolLipiHeavy`,
`AnmolLipi,Bold` and `AnmolLipi`, all WinAnsi without Unicode maps. These
are **directly converted to Unicode**, retaining dates/time values as ASCII
digits and preserving row/column spacing. This avoids OCR guessing names
or dropping headings. Cached pages record `extraction_method:
"legacy-font-conversion"` and `source_fonts`. Whole-page conversion is
used only when every PDF font belongs to the supported families, so an
English font cannot be silently transliterated as Punjabi.

Readable Unicode text remains the first choice. PDFs with scanned pages,
unknown/mixed fonts or unusable glyphs fall back to **Punjabi + English
Tesseract OCR**, if installed:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-pan tesseract-ocr-eng
```

Replacement characters, unmapped control glyphs and private-use characters
also trigger fallback. Rendering/OCR processes
one page at a time at 300 DPI, with one OCR thread, per-command timeouts
and a 31-page limit. Both overall and Punjabi-word weighted mean confidence
must be at least 60; otherwise the previous JSON stays intact. Cached OCR pages
include `extraction_method: "ocr"` and `ocr_mean_confidence`. Confidence is
an OCR diagnostic, not a measured accuracy percentage: check names and
times against the source PDF. OCR may miss headings or misread text.

For the error `PDF has no reliably extracted Punjabi text`, pull the latest
branch, update Python requirements and rerun the updater. The confirmed
SGPC font path does not require OCR packages. If a different publication
requires OCR, missing language data now produces an installation instruction
instead of the old generic rejection. The existing cron entry does not change.

Install the line in [deploy/kirtan-duties.crontab](deploy/kirtan-duties.crontab)
using the **application owner's** `crontab -e`, preserving existing jobs:

```cron
45 18 * * * /var/www/gurbani-asr/.venv/bin/python /var/www/gurbani-asr/scripts/update_kirtan_duties.py >> /var/www/gurbani-asr/data/kirtan_duties.log 2>&1
```

This is **00:15 India time every night** when the cron daemon uses UTC.
Verify the production timezone with `timedatectl` before installing. If
cron uses Asia/Kolkata, use `15 0 * * *` instead. Adjust paths for another
checkout. A user crontab has no username column. Verify installation with
`crontab -l`; repository changes alone do not activate the production job.
Rotate the duties log through the server's normal log rotation policy.

Cloud verification used a real, generated Unicode Punjabi PDF through
`pdftotext`, plus controlled HTTP responses for initial, changed, unchanged,
embedded-PDF, not-modified and failed updates. Tests also verify atomic
write failure and concurrent-run protection. Run:

```bash
.venv/bin/python -m unittest discover -s tests -p test_kirtan_duties.py -v
```

The original updater passed 146 Python tests, including 13 duties tests.
The follow-up tests the actual font list and 17 dated rows provided by the
user, including the October 15 names, dates, time headings and column order.
A generated PDF verifies direct legacy conversion without calling OCR,
and confirms unchanged cache bytes/mtime. It also tests real synthetic non-Unicode-font and scanned
PDFs, recovering all three expected names/time slots and preserving the
JSON bytes/mtime on repeat. Its live SGPC PDF remains unverified. See
[validation/kirtan-duties/summary.json](validation/kirtan-duties/summary.json)
for the original result and
[validation/kirtan-duties/ocr-summary.json](validation/kirtan-duties/ocr-summary.json)
for the follow-up: **156 tests passed, zero skipped**, including 23 duties
tests. The real OCR test requires the Punjabi and English language packs;
it explicitly skips when those tools are unavailable.

The cloud proxy currently blocks `hs.sgpc.net`. The hostname was added to
the saved environment network draft, preserving other hosts and presets;
review/save and publish that setting to permit the live-source check.
The user-provided font listing and extracted text have been verified with
the converter. The full PDF has **not** been downloaded/verified against
the live endpoint in this cloud task. No complete duties JSON is fabricated
or committed from that partial excerpt. The cloud image also has no `crontab` binary, and this task has
no shell on the production server, so the nightly production entry must
be installed there using the command above.

The structured-format update is validated against the complete JSON
uploaded by the user, with **17 dates and 255 assignments**. All **165
Python tests** pass, including 32 duties tests. They verify October 2's
15 exact times/names, noon/PM handling, paired performances, open-ended
samapti, changed heading times, rejected incomplete/conflicting/overlapping
tables, unchanged JSON timestamps, same-PDF/304 migrations and recovery
after a metadata/main-file write failure. Actual OCR checks still run
against synthetic fixtures; they do not constitute a fresh SGPC download.
See [structured validation](validation/kirtan-duties/structured-summary.json)
and [converted user JSON](validation/kirtan-duties/structured-duties.json).
