"""Download the lab's PDF and web page once, into data/downloads/.

Aim
    Save local copies of the two sources the Documents and RAG sections use,
    so later scripts work offline and always see the same content.

What you'll see
    One [OK] line per file with its size, or a [FAIL] line with a fallback.

What to notice
    Files already downloaded are skipped; use --force to download again.

Run:  uv run python scripts/00_fetch_data.py           (skip files already there)
      uv run python scripts/00_fetch_data.py --force   (download again)
"""


import argparse
import sys
import urllib.error

from lclab.data import PAPER_PATH, PAPER_URL, WEB_PATH, WEB_URL, download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args()

    failed = 0
    for url, path in [(PAPER_URL, PAPER_PATH), (WEB_URL, WEB_PATH)]:
        try:
            download(url, path, force=args.force)
        except (urllib.error.URLError, TimeoutError) as error:
            # Report the problem and keep going with the next file.
            failed += 1
            print(f"[FAIL] {path.name}  <- {url}\n       {error}")
            print(f"       Fallback: open the URL in your browser and save it as {path}")
            continue
        size_kb = path.stat().st_size / 1024
        print(f"[OK]   {path.name}  ({size_kb:.0f} KB)  <- {url}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
