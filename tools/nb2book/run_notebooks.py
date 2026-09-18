from pathlib import Path
import sys
import time
import logging
import argparse
from typing import List

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

logger = logging.getLogger(__name__)


def discover_notebooks(nbs_dir: Path) -> List[Path]:
    """Sorted list of notebooks in nbs_dir, excluding scratch/checkpoint files."""
    return sorted(
        p for p in nbs_dir.glob("*.ipynb")
        if p.name != "Untitled.ipynb" and ".ipynb_checkpoints" not in str(p)
    )


def clear_notebook_outputs(nb: nbformat.NotebookNode) -> None:
    """Clear all code cell outputs and execution counts, in place."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None


def inject_path_cell(repo_root: Path) -> nbformat.NotebookNode:
    """A transient cell that puts repo_root on sys.path for execution."""
    source = f"import sys; sys.path.insert(0, {str(repo_root)!r})"
    return nbformat.v4.new_code_cell(source=source)


def refresh_notebook(path: Path, repo_root: Path, timeout: int, clear_first: bool) -> float:
    """Execute a notebook in place, writing outputs back to disk. Returns elapsed seconds."""
    nb = nbformat.read(path, as_version=4)

    if clear_first:
        clear_notebook_outputs(nb)

    nb.cells.insert(0, inject_path_cell(repo_root))
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")

    start = time.time()
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    elapsed = time.time() - start

    del nb.cells[0]
    nbformat.write(nb, path)
    return elapsed


def main(
    project_dir: str = "",
    timeout: int = 600,
    clear_outputs: bool = True,
    build_book: bool = False,
    run_latex: bool = True,
    clean_tmp_files: bool = False,
    full_passes: bool = True,
) -> None:
    """Re-execute every notebook in nbs/ in place, optionally chaining into make_book."""
    project_path = Path(project_dir) if project_dir else Path.cwd()
    nbs_dir = project_path / "nbs"
    notebooks = discover_notebooks(nbs_dir)

    logger.info(f"Refreshing {len(notebooks)} notebooks in {nbs_dir}")

    timings = {}
    failures = []
    for nb_path in notebooks:
        logger.info(f"Executing {nb_path.name}")
        try:
            elapsed = refresh_notebook(nb_path, project_path, timeout, clear_outputs)
            timings[nb_path.stem] = round(elapsed / 60, 2)
            logger.info(f"  {nb_path.stem} done in {timings[nb_path.stem]} min")
        except Exception:
            logger.exception(f"  {nb_path.stem} failed")
            failures.append(nb_path.stem)

    logger.info(f"Timing summary (minutes): {timings}")

    if failures:
        raise RuntimeError(f"{len(failures)} notebook(s) failed: {failures}")

    if build_book:
        sys.path.insert(0, str(project_path))
        from tools.nb2book.make_book import main as make_book_main
        make_book_main(project_dir=project_dir, run_latex=run_latex, clean_tmp_files=clean_tmp_files, full_passes=full_passes)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Re-execute every notebook in nbs/ in place, so make_book can build from fresh outputs"
    )
    parser.add_argument("--project-dir", help="Project directory path", default="")
    parser.add_argument("--timeout", help="Per-notebook execution timeout in seconds", type=int, default=600)
    parser.add_argument("--clear-outputs", action=argparse.BooleanOptionalAction, default=True,
                         help="Clear existing outputs before re-executing each notebook")
    parser.add_argument("--build-book", action=argparse.BooleanOptionalAction, default=False,
                         help="Chain into make_book.py after refreshing all notebooks")
    parser.add_argument("--run-latex", action=argparse.BooleanOptionalAction, default=True,
                         help="Passed through to make_book.py when --build-book is set")
    parser.add_argument("--clean-tmp-files", action=argparse.BooleanOptionalAction, default=False,
                         help="Passed through to make_book.py when --build-book is set")
    parser.add_argument("--full-passes", action=argparse.BooleanOptionalAction, default=True,
                         help="Passed through to make_book.py when --build-book is set")

    args = parser.parse_args()
    main(
        project_dir=args.project_dir,
        timeout=args.timeout,
        clear_outputs=args.clear_outputs,
        build_book=args.build_book,
        run_latex=args.run_latex,
        clean_tmp_files=args.clean_tmp_files,
        full_passes=args.full_passes,
    )
