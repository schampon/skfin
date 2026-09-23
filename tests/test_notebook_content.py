"""Static lint checks on notebook cell source (not outputs)."""

import nbformat
import pytest

from tests.conftest import SKFIN_NBS_DIR


def _discover_notebooks():
    return sorted(
        p for p in SKFIN_NBS_DIR.glob("*.ipynb")
        if p.name != "Untitled.ipynb" and ".ipynb_checkpoints" not in str(p)
    )


@pytest.mark.parametrize("nb_path", _discover_notebooks(), ids=lambda p: p.name)
def test_no_nbsp_in_cell_source(nb_path):
    """Non-breaking spaces (U+00A0) in markdown/code source break pandoc heading
    recognition and silently corrupt the generated book's LaTeX (see make_book.py's
    convert_markdown_sections). Cell outputs are not checked: NBSP there can be
    genuine scraped data content, not an authoring artifact.
    """
    nb = nbformat.read(nb_path, as_version=4)
    offending_cells = [i for i, cell in enumerate(nb.cells) if "\xa0" in cell.source]
    assert not offending_cells, (
        f"{nb_path.name}: non-breaking space (U+00A0) found in cell source at "
        f"cell index(es) {offending_cells}. Replace with a regular space."
    )
