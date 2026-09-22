from skfin.text import green_text, red_text, color_text, highlight_lexica

LEXICA = {"positive": {"good", "strong"}, "negative": {"bad", "weak"}}


def test_green_text():
    assert "<font color='green'>hello</font>" in green_text("hello")


def test_red_text():
    assert "<font color='red'>hello</font>" in red_text("hello")


def test_color_text_positive():
    assert "green" in color_text("Good", LEXICA)


def test_color_text_negative():
    assert "red" in color_text("Bad", LEXICA)


def test_color_text_neutral():
    assert color_text("neutral", LEXICA) == "neutral"


def test_highlight_lexica_mixed():
    result = highlight_lexica("good thing bad thing", LEXICA)
    assert "green" in result
    assert "red" in result
    assert "thing" in result


def test_highlight_lexica_list_input():
    result = highlight_lexica(["good day"], LEXICA)
    assert "green" in result
