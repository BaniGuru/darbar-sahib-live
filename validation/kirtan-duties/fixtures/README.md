The two PDF files are **synthetic test documents**, not SGPC publications.
Their dates, names and time slots must not be used as actual duties.

`sgpc-pdffonts.txt` and `sgpc-extracted-excerpt.txt` retain the font listing
and first 17 dated rows pasted by the user from the real SGPC PDF. They are
an incomplete excerpt, not a replacement for the full published schedule.
They verify the supported font families, Punjabi names, diacritics and
preservation of numeric dates/times and table columns during conversion.

The source was rendered using Chromium and the system Noto Sans Gurmukhi
font. `synthetic-unmapped-font.pdf` removes its ToUnicode references,
retaining its visible glyphs while breaking direct Unicode extraction.
`synthetic-scanned.pdf` embeds a 150-DPI raster of the same page, with no
extractable text. The tests render these at 300 DPI and use Tesseract
`pan+eng`. They must recover all three names/time slots; OCR headings are
not assumed complete. No browser or font-generation tooling is needed to
replay the checked-in PDF fixtures.

These fixtures verify the OCR fallback and unchanged-file policy. They
cannot establish accuracy on the current SGPC publication, which the
cloud proxy still blocks.
