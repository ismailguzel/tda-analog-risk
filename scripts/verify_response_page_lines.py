"""Verify every response-letter page/line citation against the rendered manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT = ROOT / "manuscript" / "revised_source" / "main.pdf"
DEFAULT_RESPONSE = ROOT / "response" / "final" / "response_to_editor_and_reviewers.tex"

PAGE_LINE_RE = re.compile(
    r"\b(?P<prefix>pp?)\.~(?P<first>\d+)(?:--(?P<last>\d+))?,\s*"
    r"lines?\s+(?P<ranges>\d+--\d+(?:\s+and\s+\d+--\d+)*)"
)
LINE_TOKEN_RE = re.compile(r"^\s*(\d{1,3})\s{2,}")


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def page_line_map(pdf: Path) -> dict[int, set[int]]:
    try:
        probe = subprocess.run(
            ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        fail(f"Cannot inspect manuscript PDF with pdfinfo: {exc}")
    match = re.search(r"^Pages:\s+(\d+)", probe.stdout, flags=re.MULTILINE)
    if not match:
        fail("Manuscript PDF has no readable page count")
    page_count = int(match.group(1))

    candidates: dict[int, set[int]] = {}
    for page in range(1, page_count + 1):
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", "-f", str(page), "-l", str(page), str(pdf), "-"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            fail(f"Cannot extract manuscript page {page}: {exc}")
        numbers = {
            int(match.group(1))
            for line in result.stdout.splitlines()
            if (match := LINE_TOKEN_RE.match(line))
        }
        candidates[page] = numbers

    # Printed line numbers occur in consecutive runs. This removes unrelated
    # table/axis numbers that happen to begin a layout-preserved text line.
    return {
        page: {
            number
            for number in numbers
            if number - 1 in numbers or number + 1 in numbers
        }
        for page, numbers in candidates.items()
    }


def references(source: str) -> list[tuple[int, int, int, int, str]]:
    if "p.\\~" in source or "pp.\\~" in source:
        fail("Response letter contains forbidden p.\\~ or pp.\\~ spacing")
    found: list[tuple[int, int, int, int, str]] = []
    for match in PAGE_LINE_RE.finditer(source):
        first_page = int(match.group("first"))
        last_page = int(match.group("last") or match.group("first"))
        if first_page > last_page:
            fail(f"Invalid page range in response citation: {match.group(0)}")
        for line_start, line_end in re.findall(r"(\d+)--(\d+)", match.group("ranges")):
            start, end = int(line_start), int(line_end)
            if start > end:
                fail(f"Invalid line range in response citation: {match.group(0)}")
            found.append((first_page, last_page, start, end, match.group(0)))
    if not found:
        fail("No page/line citations were found in the response letter")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    args = parser.parse_args()
    source = args.response.read_text(encoding="utf-8")
    line_map = page_line_map(args.manuscript)
    refs = references(source)
    for first_page, last_page, start, end, citation in refs:
        if start not in line_map.get(first_page, set()):
            fail(f"Line {start} is not on first cited page {first_page}: {citation}")
        if end not in line_map.get(last_page, set()):
            fail(f"Line {end} is not on last cited page {last_page}: {citation}")
    print(f"[OK] Response page/line references verified: {len(refs)} references")


if __name__ == "__main__":
    main()
