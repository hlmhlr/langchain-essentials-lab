"""Tests for lclab.data.download, without touching the internet."""

from lclab import data


def test_download_skips_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "file.txt"
    path.write_text("already here")

    # If download tried the network, this would raise and fail the test.
    def no_network(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(data.urllib.request, "urlopen", no_network)
    assert data.download("https://example.com/x", path) == path
    assert path.read_text() == "already here"


def test_paths_are_inside_the_project():
    assert data.PAPER_PATH.parent == data.DOWNLOAD_DIR
    assert data.DOWNLOAD_DIR.is_relative_to(data.ROOT)
