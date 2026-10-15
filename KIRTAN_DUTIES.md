The nightly job fetches the Punjabi kirtan duties from
<https://hs.sgpc.net/kirtan_duty_punjabi_pdf.php> and saves
`data/kirtan_duties.json`. It runs separately from the live ASR service.

The JSON retains the PDF's page numbers, headings, dates and duty rows as
Unicode Punjabi text. Each row has `text` and `columns` split at layout
spacing. Columns remain in PDF order; they are not assigned guessed names
or interpreted as calendar dates. Source URLs, fetch time, PDF/content
SHA-256 and HTTP validators are included for provenance.

New duties replace the JSON atomically. Unchanged PDF bytes, HTTP 304, or
regenerated PDFs with the same extracted text leave its **contents and
modification time unchanged**. Timeouts, HTTP errors, invalid HTML/PDFs,
missing extraction tools and unreadable/scanned/legacy-font Punjabi PDFs
retain the last good JSON and exit nonzero. Concurrent runs are locked.
A failed first fetch creates no placeholder duties JSON. Generated duties,
lock and log files are ignored by Git.

On the production server, install Poppler and cron if absent:

```bash
sudo apt-get install poppler-utils cron
cd /var/www/gurbani-asr
.venv/bin/python scripts/update_kirtan_duties.py
```

The script uses the existing `httpx` Python dependency. The application
owner must be able to write `data/`. Its default output is resolved from
the script location, so cron does not need a working directory. An
alternative location can be supplied with `--output /path/to/duties.json`.

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

All 146 Python tests passed, including 13 duties updater tests. See
[validation/kirtan-duties/summary.json](validation/kirtan-duties/summary.json).

The cloud proxy currently blocks `hs.sgpc.net`. The hostname was added to
the saved environment network draft, preserving other hosts and presets;
review/save and publish that setting to permit the live-source check.
The exact SGPC PDF layout and its font encoding have **not** been verified
against the live endpoint yet. No real duties data is fabricated or
committed. The cloud image also has no `crontab` binary, and this task has
no shell on the production server, so the nightly production entry must
be installed there using the command above.
