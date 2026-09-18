from pathlib import Path

import nbformat

from tools.nb2book.run_notebooks import (
    clear_notebook_outputs,
    discover_notebooks,
    inject_path_cell,
)


def test_discover_notebooks_sorted_and_filtered(tmp_path):
    for name in ["b.ipynb", "a.ipynb", "Untitled.ipynb"]:
        (tmp_path / name).write_text("{}")
    checkpoints_dir = tmp_path / ".ipynb_checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "a-checkpoint.ipynb").write_text("{}")

    result = discover_notebooks(tmp_path)
    assert result == [tmp_path / "a.ipynb", tmp_path / "b.ipynb"]


def test_discover_notebooks_empty_dir(tmp_path):
    assert discover_notebooks(tmp_path) == []


def test_clear_notebook_outputs_clears_code_cells_only():
    nb = nbformat.v4.new_notebook()
    code_cell = nbformat.v4.new_code_cell(source="1 + 1")
    code_cell.outputs = [nbformat.v4.new_output(output_type="stream", text="2")]
    code_cell.execution_count = 3
    markdown_cell = nbformat.v4.new_markdown_cell(source="# Title")
    nb.cells = [markdown_cell, code_cell]

    clear_notebook_outputs(nb)

    assert nb.cells[1].outputs == []
    assert nb.cells[1].execution_count is None
    assert nb.cells[0].source == "# Title"


def test_inject_path_cell_contains_repo_root():
    repo_root = Path("/home/user/dev/projects/skfin2")
    cell = inject_path_cell(repo_root)
    assert cell.cell_type == "code"
    assert str(repo_root) in cell.source
    assert "sys.path.insert" in cell.source
