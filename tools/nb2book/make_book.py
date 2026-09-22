from pathlib import Path
import sys
import shutil
import subprocess
import logging
import argparse
import json
import re
from typing import List


logger = logging.getLogger(__name__)

FOOTNOTE_TAG = "#_footnote"
_FOOTNOTE_START = "ZZFOOTNOTESTARTZZ"
_FOOTNOTE_END = "ZZFOOTNOTEENDZZ"


def remove_hide_cells(content: str, pattern: str = "sh{} hide") -> str:
    """Remove hidden cells from LaTeX content."""
    while pattern in content:
        start_idx = content.rfind("begin{tcolorbox}", 0, content.find(pattern))
        if start_idx == -1:
            break
        end_idx = content.find("end{tcolorbox}", start_idx) + len("end{tcolorbox}")
        content = content[:start_idx-1] + content[end_idx+1:]
    return content


def convert_markdown_sections(content: str, level: int = 2) -> str:
    """Convert markdown-style headers to LaTeX sections."""
    patterns = {
        1: (r"##~", r"\\section{"),
        2: (r"###~", r"\\subsection{")
    }

    if level not in patterns:
        return content

    pattern, replacement = patterns[level]
    return re.sub(rf"{pattern}(.*?)\\n", rf"{replacement}\1}}", content)


def stage_footnote_tagged_notebooks(notebook_dir: Path, staging_dir: Path) -> None:
    """Copy notebooks to staging_dir, marking cells tagged with FOOTNOTE_TAG.

    Tag a markdown cell as a footnote by starting it with a line containing
    only "#_footnote" -- the whole rest of that cell's content becomes the
    footnote body. The tag is stripped and the remaining content wrapped in
    internal sentinels here (before nbconvert runs) since cell boundaries
    aren't otherwise recoverable from the flattened LaTeX text that
    convert_footnote_cells operates on. Original notebooks are untouched.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    for nb_path in notebook_dir.glob("*.ipynb"):
        nb = json.loads(nb_path.read_text())
        for cell in nb["cells"]:
            if cell.get("cell_type") != "markdown":
                continue
            source = "".join(cell["source"])
            stripped = source.lstrip()
            if stripped.startswith(FOOTNOTE_TAG):
                body = stripped[len(FOOTNOTE_TAG):].lstrip("\n")
                cell["source"] = [f"{_FOOTNOTE_START}\n{body}\n{_FOOTNOTE_END}"]
        (staging_dir / nb_path.name).write_text(json.dumps(nb))


def convert_footnote_cells(content: str) -> str:
    """Convert markdown cells tagged as footnotes into \\footnote{...} calls.

    Matches the internal sentinels injected by stage_footnote_tagged_notebooks
    for cells tagged with FOOTNOTE_TAG ("#_footnote"). The leading blank line
    (paragraph break) that nbconvert puts before the tagged cell is swallowed
    too, so the \\footnote{} attaches to the end of the preceding sentence
    instead of starting an empty paragraph of its own; the blank line after
    it is left alone so whatever follows still starts its own paragraph.
    nbconvert indents wrapped lines with leading spaces, so what precedes
    the sentinel is a run of whitespace, not just newlines.

    Avoid fenced code blocks (```) inside a tagged cell: they become a
    Verbatim environment, which LaTeX cannot place inside \\footnote{} and
    corrupts the rest of the document. Use inline `code` spans instead.
    """
    pattern = re.compile(rf"\s+{_FOOTNOTE_START}(.*?){_FOOTNOTE_END}", re.DOTALL)
    return pattern.sub(lambda m: "\\footnote{" + m.group(1).strip() + "}", content)


def process_side_by_side_images(content: str, pattern: str = "sh{} sidebyside") -> str:
    """Process side-by-side image layouts."""
    while pattern in content:
        start_idx = content.find(pattern)
        if start_idx == -1:
            break

        center_end = content.find("\\n    \\end{center}", start_idx)
        if center_end == -1:
            break

        adjust_start = content.find("adjustimage", center_end)
        if adjust_start == -1:
            break

        content = content[:center_end] + content[adjust_start-1:]

        # Find next occurrence
        next_idx = content.find(pattern, start_idx + len(pattern))
        if next_idx == -1:
            break

    return content


def convert_notebooks_to_latex(notebook_dir: Path, latex_dir: Path) -> None:
    """Convert Jupyter notebooks to LaTeX files."""
    latex_dir.mkdir(parents=True, exist_ok=True)

    staging_dir = latex_dir / "_staged_nbs"
    stage_footnote_tagged_notebooks(notebook_dir, staging_dir)

    cmd = [
        "jupyter", "nbconvert", "--to", "latex",
        str(staging_dir / "*.ipynb"),
        "--output-dir", str(latex_dir)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert notebooks: {e}")
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def get_latex_files(latex_dir: Path) -> List[str]:
    """Get sorted list of LaTeX files, excluding temporary and book files."""
    return sorted([
        f.name for f in latex_dir.iterdir()
        if f.suffix == ".tex" and "tmp" not in f.name and "book" not in f.name
    ])


def process_latex_content(content: str) -> str:
    """Apply all LaTeX content transformations."""
    start_idx = content.find("\\maketitle")
    end_idx = content.find("\\end{document}")

    if start_idx == -1 or end_idx == -1:
        return content

    # Extract main content
    main_content = content[start_idx + 10:end_idx]

    main_content = re.sub(r'\bsubsection\s*{\s*([A-Za-z0-9]+)\*\s*}', r'subsection*{\1}', main_content)
    # Apply transformations
    transformations = [
        ("\\section", "\\chapter"),
        ("\\subsection", "\\section"),
        ("\\subsubsection", "\\subsection"),
        ("\\subsubsubsection", "\\subsubsection"),
        ("\\prompt{Out}{outcolor}{5}{}", ""),
        ("0.9\\paperheight", "0.25\\paperheight")
    ]

    for old, new in transformations:
        main_content = main_content.replace(old, new)

    # Apply custom processing functions
    main_content = process_side_by_side_images(main_content)
    main_content = remove_hide_cells(main_content)
    main_content = convert_footnote_cells(main_content)
    main_content = convert_markdown_sections(main_content, level=2)
    main_content = convert_markdown_sections(main_content, level=1)

    return main_content


def create_book_latex(latex_dir: Path, template_dir: Path, include_files: List[str]) -> None:
    """Create the main book LaTeX file."""
    # Copy references
    refs_src = template_dir / "references.bib"
    refs_dst = latex_dir / "references.bib"
    if refs_src.exists():
        shutil.copy(refs_src, refs_dst)

    # Read template
    template_file = template_dir / "book_template.tex"
    with open(template_file, "r") as f:
        template_content = f.read()


    # Replace include placeholder
    insert_string = []
    for name in include_files:
        insert_string += [r"\include{" + name + "}"]

    include_statements = " \n ".join(insert_string)
    book_content = template_content.replace("%INCLUDE_HERE", include_statements)

    # Write book file
    with open(latex_dir / "book.tex", "w") as f:
        f.write(book_content)


def compile_latex(latex_dir: Path, book_dir: Path, clean_tmp_files: bool = False, full_passes: bool = True) -> None:
    """Compile LaTeX to PDF."""
    try:
        # First pass: writes .aux/.toc, needed before bibtex and before the ToC can be typeset
        subprocess.run(["pdflatex", "book.tex"], cwd=latex_dir, check=True)

        # Run bibtex
        subprocess.run(["bibtex", "book"], cwd=latex_dir, check=True)

        if full_passes:
            # Two more passes: first typesets the table of contents and citations from
            # the .toc/.bbl written above, second stabilizes any resulting page-number shifts
            subprocess.run(["pdflatex", "book.tex"], cwd=latex_dir, check=True)
            subprocess.run(["pdflatex", "book.tex"], cwd=latex_dir, check=True)

        # Copy PDF to book directory
        pdf_src = latex_dir / "book.pdf"
        pdf_dst = book_dir / "book.pdf"
        if pdf_src.exists():
            shutil.copy(pdf_src, pdf_dst)

        # Clean temporary files
        if clean_tmp_files:
            for tmp_file in latex_dir.glob("tmp*.*"):
                tmp_file.unlink()

    except subprocess.CalledProcessError as e:
        logger.error(f"LaTeX compilation failed: {e}")
        raise


def main(project_dir: str = '', run_latex: bool = True, clean_tmp_files: bool = False, full_passes: bool = True) -> None:
    """Convert Jupyter notebooks to LaTeX and compile them into a book PDF."""
    # Setup directories
    project_path = Path(project_dir) if project_dir else Path.cwd()
    notebook_dir = project_path / "nbs"
    book_dir = project_path / "book"
    template_dir = project_path / "template"
    latex_dir = book_dir / "tex"

    # Convert notebooks to LaTeX
    convert_notebooks_to_latex(notebook_dir, latex_dir)

    # Get and process LaTeX files
    filenames = get_latex_files(latex_dir)
    logger.info(f"Processing files: {filenames}")

    include_files = []
    for i, filename in enumerate(filenames):
        # Read and process file
        with open(latex_dir / filename, "r") as f:
            content = f.read()

        processed_content = process_latex_content(content)

        # Write temporary file
        tmp_filename = f"tmp{i}"
        with open(latex_dir / f"{tmp_filename}.tex", "w") as f:
            f.write(processed_content)

        include_files.append(tmp_filename)

    # Create book LaTeX file
    create_book_latex(latex_dir, template_dir, include_files)

    # Compile if requested
    if run_latex:
        compile_latex(latex_dir, book_dir, clean_tmp_files, full_passes)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(description="Make a book from a project directory")
    parser.add_argument("--project-dir", help="Project directory path", default='')
    parser.add_argument("--run-latex", action=argparse.BooleanOptionalAction, default=True,
                         help="Run LaTeX compilation")
    parser.add_argument("--clean-tmp-files", action=argparse.BooleanOptionalAction, default=False,
                         help="Clean temporary files")
    parser.add_argument("--full-passes", action=argparse.BooleanOptionalAction, default=True,
                         help="Run the extra pdflatex passes needed for the table of contents and citations")

    args = parser.parse_args()
    main(
        project_dir=args.project_dir,
        run_latex=args.run_latex,
        clean_tmp_files=args.clean_tmp_files,
        full_passes=args.full_passes,
    )
