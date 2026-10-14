from gateway.orchestration.normalize import normalize_for_speech, strip_reasoning


def test_markdown_is_stripped():
    text = "# Heading\n\n- **bold** item\n- *italic* item\n1. numbered"
    out = normalize_for_speech(text)
    assert "#" not in out and "*" not in out and "- " not in out
    assert "bold item" in out


def test_links_and_urls_become_speakable():
    out = normalize_for_speech("See [the docs](https://example.com/x) or https://a.io/b?q=1 today.")
    assert "https://" not in out
    assert "the docs" in out
    assert "a link" in out


def test_code_fence_is_summarized():
    out = normalize_for_speech("Run this:\n```python\nprint('hi')\n```\nThen continue.")
    assert "print" not in out
    assert "code omitted" in out


def test_citations_and_emoji_removed():
    out = normalize_for_speech("Great point [1, 2] 🎉🚀 indeed!!!")
    assert "[1" not in out
    assert "🎉" not in out
    assert out.endswith("indeed!")


def test_dangling_think_is_never_spoken():
    assert normalize_for_speech("Before. <think>secret partial reasoning") == "Before."
    assert strip_reasoning("<think>a</think>Answer") == "Answer"


def test_stt_annotations_are_stripped():
    from gateway.orchestration.normalize import strip_stt_annotations

    # Pure non-speech markers -> empty: no turn should start.
    assert strip_stt_annotations("[BLANK_AUDIO]") == ""
    assert strip_stt_annotations(" [ Silence ] ") == ""
    assert strip_stt_annotations("(wind blowing) *coughs* ♪♪") == ""
    # Mixed content keeps the real words.
    assert strip_stt_annotations("[music] hello there (laughs)") == "hello there"
    # Plain speech is untouched, punctuation included.
    assert strip_stt_annotations("What time is it?") == "What time is it?"
    assert strip_stt_annotations("[BLANK_AUDIO].") == ""


def test_stage_directions_are_never_spoken_or_shown():
    from gateway.orchestration.normalize import (
        StageTagStreamFilter,
        strip_stage_tags,
    )

    out = normalize_for_speech("Gandalf falls in Moria. [pause] He returns as the White.")
    assert "[pause]" not in out and "pause" not in out
    assert "Gandalf falls in Moria." in out

    assert strip_stage_tags("Homer, Marge. [pause] Each episode. [sighs softly]") == (
        "Homer, Marge. Each episode."
    )
    # Numeric brackets and long spans are not stage cues.
    assert strip_stage_tags("see [12] for details") == "see [12] for details"

    # Streaming filter handles tags split across deltas.
    f = StageTagStreamFilter()
    got = f.feed("Sauron falls. [pau") + f.feed("se] Peace returns.") + f.flush()
    assert got == "Sauron falls.  Peace returns."
    # An unclosed bracket that can't be a tag passes through on flush.
    f = StageTagStreamFilter()
    got = f.feed("math: a[i") + f.feed("] = 3, done")
    assert "a[i]" in got


def test_em_dash_becomes_pause():
    out = normalize_for_speech("It works—mostly—on the first try.")
    assert "—" not in out
    assert "," in out
