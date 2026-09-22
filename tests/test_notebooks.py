"""Notebook execution tests — run each notebook end-to-end as a regression baseline."""

from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from tests.conftest import SKFIN2_NBS_DIR, SKFIN2_ROOT

NOTEBOOKS = sorted(
    p
    for p in SKFIN2_NBS_DIR.glob("*.ipynb")
    if p.name != "Untitled.ipynb" and ".ipynb_checkpoints" not in str(p)
)


def _inject_path_cell(repo_root: Path) -> nbformat.NotebookNode:
    source = f"import sys; sys.path.insert(0, {str(repo_root)!r})"
    return nbformat.v4.new_code_cell(source=source)


@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOKS,
    ids=[p.stem for p in NOTEBOOKS],
)
def test_notebook_execution(notebook_path: Path, nb_timeout: int):
    nb = nbformat.read(notebook_path, as_version=4)
    nb.cells.insert(0, _inject_path_cell(SKFIN2_ROOT))
    ep = ExecutePreprocessor(timeout=nb_timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(SKFIN2_NBS_DIR)}})
