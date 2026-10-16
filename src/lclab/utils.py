"""Small printing helpers shared by the scripts."""


def print_header(title: str) -> None:
    """Print a section title with a line under it."""
    print(f"\n{title}\n{'=' * len(title)}")


def print_block(label: str, text: str) -> None:
    """Print a labelled piece of text, e.g. a prompt or a model answer."""
    print(f"\n--- {label} ---\n{text.strip()}")
