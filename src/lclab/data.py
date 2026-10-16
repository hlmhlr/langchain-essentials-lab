"""Where the lab's source documents come from and where they are saved.

The lab loads a PDF and a web page straight from the internet every time.
Here they are downloaded once into data/downloads/, so the later scripts
work offline and always see the same content.
"""

import ssl
import urllib.request
from pathlib import Path

import certifi

# Project root = two folders up from this file (src/lclab/data.py).
ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_DIR = ROOT / "data" / "downloads"

# The paper used in the lab: "Revolutionizing Mental Health Care through
# LangChain" (arXiv:2403.05568), hosted by IBM Skills Network.
PAPER_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "96-FDF8f7coh0ooim7NyEQ/langchain-paper.pdf"
)
PAPER_PATH = DOWNLOAD_DIR / "langchain-paper.pdf"

# The lab's page (python.langchain.com/v0.2/docs/introduction/) now redirects
# to the agents overview. The Retrieval page covers what Exercises 3-4 ask about.
WEB_URL = "https://docs.langchain.com/oss/python/langchain/retrieval"
WEB_PATH = DOWNLOAD_DIR / "langchain-retrieval.html"


# Check HTTPS certificates against certifi's up-to-date list of trusted
# authorities. The Windows certificate store can be out of date, which makes
# some sites fail with "certificate has expired" even though browsers work.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def download(url: str, path: Path, force: bool = False) -> Path:
    """Download `url` to `path`, unless the file is already there."""
    if path.exists() and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "lclab/0.1"})
    with urllib.request.urlopen(request, timeout=30, context=SSL_CONTEXT) as response:
        path.write_bytes(response.read())
    return path
