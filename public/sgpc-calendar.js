/*
MIT License

Copyright (c) 2023 Janpreet Singh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// SGPC modified Nanakshahi: published Jantri month starts, not fixed dates.
// Data: nanakshahi-jantri 1.2.0 (Janpreet Singh, MIT); see SGPC_CALENDAR.md.
const monthStarts = [
  [
    "2025-03-14",
    "ਚੇਤ",
    557
  ],
  [
    "2025-04-13",
    "ਵੈਸਾਖ",
    557
  ],
  [
    "2025-05-14",
    "ਜੇਠ",
    557
  ],
  [
    "2025-06-15",
    "ਹਾੜ",
    557
  ],
  [
    "2025-07-16",
    "ਸਾਵਣ",
    557
  ],
  [
    "2025-08-16",
    "ਭਾਦੋਂ",
    557
  ],
  [
    "2025-09-16",
    "ਅੱਸੂ",
    557
  ],
  [
    "2025-10-17",
    "ਕੱਤਕ",
    557
  ],
  [
    "2025-11-16",
    "ਮੱਘਰ",
    557
  ],
  [
    "2025-12-15",
    "ਪੋਹ",
    557
  ],
  [
    "2026-01-14",
    "ਮਾਘ",
    557
  ],
  [
    "2026-02-12",
    "ਫੱਗਣ",
    557
  ],
  [
    "2026-03-14",
    "ਚੇਤ",
    558
  ],
  [
    "2026-04-14",
    "ਵੈਸਾਖ",
    558
  ],
  [
    "2026-05-15",
    "ਜੇਠ",
    558
  ],
  [
    "2026-06-15",
    "ਹਾੜ",
    558
  ],
  [
    "2026-07-16",
    "ਸਾਵਣ",
    558
  ],
  [
    "2026-08-17",
    "ਭਾਦੋਂ",
    558
  ],
  [
    "2026-09-17",
    "ਅੱਸੂ",
    558
  ],
  [
    "2026-10-17",
    "ਕੱਤਕ",
    558
  ],
  [
    "2026-11-16",
    "ਮੱਘਰ",
    558
  ],
  [
    "2026-12-16",
    "ਪੋਹ",
    558
  ],
  [
    "2027-01-14",
    "ਮਾਘ",
    558
  ],
  [
    "2027-02-13",
    "ਫੱਗਣ",
    558
  ]
];
const coverageEnd = "2027-03-15";
const indiaDateFormatter = new Intl.DateTimeFormat("en-CA", {
  timeZone: "Asia/Kolkata", year: "numeric", month: "2-digit", day: "2-digit"
});
const indiaTimeFormatter = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit", hour12: true
});
const punjabiDigits = value => String(value).replace(/\d/g, digit => "੦੧੨੩੪੫੬੭੮੯"[Number(digit)]);

export function darbarDateTime(now = new Date()) {
  const parts = Object.fromEntries(indiaDateFormatter.formatToParts(now).map(p => [p.type, p.value]));
  const date = `${parts.year}-${parts.month}-${parts.day}`;
  const month = date < coverageEnd && monthStarts.findLast(row => row[0] <= date);
  // Hide unavailable dates instead of inventing next year's official Jantri.
  const day = month ? 1 + Math.round((Date.parse(date)-Date.parse(month[0]))/86400000) : null;
  return {
    date: month ? `${punjabiDigits(day)} ${month[1]} ${punjabiDigits(month[2])}` : "",
    time: indiaTimeFormatter.format(now).toLowerCase(),
    civilDate: date
  };
}
