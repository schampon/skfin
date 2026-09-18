from tools.nb2book.make_book import (
    convert_markdown_sections,
    process_latex_content,
    process_side_by_side_images,
    remove_hide_cells,
)


def test_remove_hide_cells_strips_hidden_box():
    content = r"before \begin{tcolorbox} sh{} hide stuff \end{tcolorbox} after"
    assert remove_hide_cells(content) == "before after"


def test_remove_hide_cells_no_pattern_unchanged():
    content = "no pattern here"
    assert remove_hide_cells(content) == content


def test_convert_markdown_sections_level1():
    content = "##~Introduction\\n rest"
    assert convert_markdown_sections(content, level=1) == "\\section{Introduction} rest"


def test_convert_markdown_sections_level2():
    content = "###~Details\\n rest"
    assert convert_markdown_sections(content, level=2) == "\\subsection{Details} rest"


def test_convert_markdown_sections_unsupported_level_unchanged():
    content = "###~Details\\n rest"
    assert convert_markdown_sections(content, level=3) == content


def test_process_side_by_side_images_collapses_block():
    content = r"a sh{} sidebyside b \begin{center}\n    \end{center} c adjustimage END d"
    assert process_side_by_side_images(content) == r"a sh{} sidebyside b \begin{center} adjustimage END d"


def test_process_side_by_side_images_no_pattern_unchanged():
    content = "no pattern"
    assert process_side_by_side_images(content) == content


def test_process_latex_content_transforms_document():
    doc = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\section{Intro}\n"
        "Some text.\n"
        "\\subsection{Sub}\n"
        "More \\prompt{Out}{outcolor}{5}{} text.\n"
        "\\includegraphics[width=0.9\\paperheight]{img.png}\n"
        "\\end{document}\n"
    )
    result = process_latex_content(doc)
    assert "\\chapter{Intro}" in result
    assert "\\section{Sub}" in result
    assert "\\prompt{Out}{outcolor}{5}{}" not in result
    assert "0.25\\paperheight" in result
    assert "0.9\\paperheight" not in result


def test_process_latex_content_missing_maketitle_unchanged():
    doc = "no maketitle here \\end{document}"
    assert process_latex_content(doc) == doc


def test_process_latex_content_missing_end_document_unchanged():
    doc = "\\maketitle but no end"
    assert process_latex_content(doc) == doc
