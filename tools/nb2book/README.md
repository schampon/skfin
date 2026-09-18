# nb2book

Two chained CLI scripts that turn the teaching notebooks in `nbs/` into a compiled PDF book.

```
run_notebooks.py --build-book
        │
        ├── re-executes every notebook in nbs/ (fresh outputs)
        │
        └── chain into make_book.py
                │
                ├── nbconvert: nbs/*.ipynb -> book/tex/*.tex
                ├── stitch chapters into book/tex/book.tex (via template/book_template.tex)
                └── pdflatex + bibtex -> book/book.pdf
```

## `run_notebooks.py`

Re-executes every notebook in `nbs/` in place, writing fresh outputs back to disk. Does **not** touch LaTeX/PDF — its only job is to refresh notebook outputs (e.g. after a code change in `skfin/`, or to pick up new data).

```bash
python tools/nb2book/run_notebooks.py --project-dir .
```

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--project-dir` | `""` (cwd) | Project root; expects a `nbs/` subdirectory |
| `--timeout` | `600` | Per-notebook execution timeout, in seconds |
| `--clear-outputs` / `--no-clear-outputs` | `True` | Clear existing outputs before re-executing (forces a real rerun rather than reusing cached cell outputs) |
| `--build-book` / `--no-build-book` | `False` | Chain into `make_book.py` once all notebooks succeed |
| `--run-latex`, `--clean-tmp-files`, `--full-passes` | see below | Passed straight through to `make_book.py` when `--build-book` is set |

Notebooks are executed with a `python3` kernel, with the project root injected onto `sys.path` for the duration of execution (so `import skfin` resolves without installing the package). If any notebook raises, execution continues through the rest, and a `RuntimeError` listing all failures is raised at the end — so one broken notebook doesn't block the others from refreshing.

## `make_book.py`

Converts whatever is currently saved in `nbs/*.ipynb` (existing outputs, not re-executed) into a single PDF book. Does **not** run any notebook code itself.

```bash
python tools/nb2book/make_book.py --project-dir .
```

Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `--project-dir` | `""` (cwd) | Project root; expects `nbs/`, `book/`, `template/` subdirectories |
| `--run-latex` / `--no-run-latex` | `True` | Actually invoke `pdflatex`/`bibtex`; if off, only generates `book/tex/*.tex` |
| `--clean-tmp-files` / `--no-clean-tmp-files` | `False` | Delete the intermediate `tmpN.tex` per-chapter files after compiling |
| `--full-passes` / `--no-full-passes` | `True` | Run the two extra `pdflatex` passes needed to typeset the table of contents and citations (see below) |

Pipeline inside `main()`:
1. `convert_notebooks_to_latex` — `jupyter nbconvert --to latex` on every notebook in `nbs/`, writing to `book/tex/`.
2. For each generated `.tex` file, `process_latex_content` extracts the body between `\maketitle` and `\end{document}` and rewrites it: promotes `\section`→`\chapter` (and one level down for everything else), strips cells tagged `sh{} hide`, collapses `sh{} sidebyside` image blocks, and converts markdown-style `##`/`###` headers that survived nbconvert's escaping into proper `\section`/`\subsection` commands. Each result is written to `book/tex/tmpN.tex`.
3. `create_book_latex` stitches the `tmpN.tex` files into `book/tex/book.tex` using `template/book_template.tex` (which owns the actual document class, fonts, and packages — the per-chapter files' own preambles are discarded).
4. `compile_latex` runs `pdflatex` → `bibtex` → (if `--full-passes`) `pdflatex` → `pdflatex`, then copies the result to `book/book.pdf`. The extra passes are required because `.toc`/`.bbl` are written on one pass and only typeset on a later one — without them the book compiles but the table of contents and citations are missing or stale.

## Full rebuild

To refresh every notebook and rebuild the PDF in one command:

```bash
cd ~/dev/projects/skfin2
python tools/nb2book/run_notebooks.py --project-dir . --build-book --clean-tmp-files
```

Notebooks that make live network calls (e.g. `46_Text_processing.ipynb`, which downloads a HuggingFace model) need `SSL_CERT_FILE` and `HF_HUB_DISABLE_XET=1` set in the executing environment/kernel beforehand.

## Notes

- `--build-book`'s deferred import (`from tools.nb2book.make_book import main as make_book_main`) requires `tools` to be importable as a package; `run_notebooks.py` inserts the project root onto `sys.path` itself before that import, so running it directly as a script (`python tools/nb2book/run_notebooks.py ...`) works without `PYTHONPATH` tricks.
- Regular pytest coverage for both scripts' pure/unit-testable logic lives in `tests/test_run_notebooks.py` and `tests/test_make_book.py`.
