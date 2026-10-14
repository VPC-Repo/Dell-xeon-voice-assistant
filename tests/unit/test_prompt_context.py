"""Deployment grounding appended to the system prompt.

The canned retail demo hinges on one property: facts the model cannot
know (this store's aisle map) reach it through the system prompt, and
their absence degrades to the plain assistant rather than a failed
session. The file is data, not code, so the loader's edge cases are the
whole feature.
"""

from __future__ import annotations

from gateway.session import build_system_prompt, load_prompt_context


def test_context_is_appended_to_both_prompt_variants(tmp_path):
    f = tmp_path / "store.txt"
    f.write_text("Store map: curry powder is on aisle 4.")
    ctx = load_prompt_context(str(f))
    for opener in (True, False):
        prompt = build_system_prompt(opener, ctx)
        assert prompt.endswith("Store map: curry powder is on aisle 4.")
        # The base conversational instructions must survive untouched.
        assert "flowing conversational prose" in prompt


def test_empty_path_means_plain_assistant():
    assert load_prompt_context("") == ""
    assert build_system_prompt(True, "") == build_system_prompt(True)


def test_missing_file_degrades_without_raising(tmp_path):
    assert load_prompt_context(str(tmp_path / "absent.txt")) == ""


def test_blank_file_adds_nothing(tmp_path):
    f = tmp_path / "blank.txt"
    f.write_text("   \n\n  ")
    assert load_prompt_context(str(f)) == ""


def test_shipped_grocery_persona_is_loadable_and_small():
    # The repo's canned demo file must stay valid and inside the prefill
    # budget the feature promises (~400 tokens ~= 1600 chars at 4/token).
    ctx = load_prompt_context("prompts/grocery-store.txt")
    assert "aisle 4" in ctx and "chickpea curry" in ctx
    assert len(ctx) < 2600, f"{len(ctx)} chars; trim the persona"


def test_session_prompt_carries_the_context(tmp_path):
    from gateway.adapters.mock import MockLLM, MockStreamingSTT, MockTTS
    from gateway.config import Settings
    from gateway.session import GenerationSlots, Session

    f = tmp_path / "store.txt"
    f.write_text("Curry powder: aisle 4, halfway down, left, top shelf.")
    settings = Settings(_env_file=None, PROMPT_CONTEXT_FILE=str(f))
    session = Session(settings, llm=MockLLM(),
                      tts=MockTTS(sample_rate=24000),
                      generation_slots=GenerationSlots(2),
                      streaming_stt=MockStreamingSTT())
    assert session.history[0].role == "system"
    assert "aisle 4, halfway down, left, top shelf" in session.history[0].content
