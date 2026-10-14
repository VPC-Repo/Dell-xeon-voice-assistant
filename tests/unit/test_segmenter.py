from gateway.orchestration.segmenter import SpeechSegmenter


def feed_all(seg: SpeechSegmenter, text: str, chunk: int = 5):
    """Simulate token streaming by feeding small chunks."""
    out = []
    for i in range(0, len(text), chunk):
        out.extend(seg.feed(text[i : i + chunk]))
    return out


def test_emits_sentence_after_minimum_length():
    seg = SpeechSegmenter()
    got = feed_all(seg, "This is a reasonably long first sentence for the test. And more")
    assert len(got) == 1
    assert got[0].text.startswith("This is a reasonably long")
    assert got[0].text.endswith(".")


def test_first_segment_emits_short_opening_sentence_immediately():
    # The prompt asks for a few-word opener so it can play while the rest
    # streams; segment 0 must not hold "Sure." hostage to length minimums
    # (measured cost: ~1s of end-of-speech -> first-audio).
    seg = SpeechSegmenter()
    got = feed_all(seg, "Sure. ")
    assert len(got) == 1
    assert got[0].segment_id == 0
    assert got[0].raw.strip() == "Sure."


def test_short_sentence_buffers_until_next_boundary():
    seg = SpeechSegmenter()
    feed_all(seg, "Here is a complete opening sentence for this reply. ")  # seg 0
    got = feed_all(seg, "Yes. ")
    assert got == []  # later segments: too short to speak alone
    got = feed_all(seg, "I can absolutely help you with that request today. ")
    assert len(got) == 1
    assert "Yes." in got[0].text


def test_decimal_is_not_a_boundary():
    seg = SpeechSegmenter()
    got = feed_all(seg, "The measured value was exactly 3.14159 units in the final test run. ")
    assert len(got) == 1
    assert "3.14159" in got[0].text


def test_abbreviation_is_not_a_boundary():
    seg = SpeechSegmenter()
    got = feed_all(seg, "You should talk with Dr. Smith about the results tomorrow morning. ")
    assert len(got) == 1
    assert "Dr. Smith" in got[0].text


def test_quoted_span_is_not_split():
    seg = SpeechSegmenter()
    text = 'He said "Stop. Wait." and then he left the room rather quickly afterwards. '
    got = feed_all(seg, text)
    assert len(got) == 1
    assert '"Stop. Wait."' in got[0].raw


def test_first_segment_hard_cap_bounds_synthesis_cost():
    seg = SpeechSegmenter()
    # A rambling unpunctuated opener must still break by ~80 chars, since
    # first-audio synthesis time scales with segment length.
    words = "the assistant considered many interesting possibilities together " * 3
    got = feed_all(seg, words)
    assert got and got[0].segment_id == 0
    assert len(got[0].raw) <= 80


def test_overlong_buffer_forces_a_break():
    seg = SpeechSegmenter(max_chars=80)
    words = "one two three four five six seven eight nine ten " * 4  # no punctuation
    got = feed_all(seg, words)
    assert got, "expected a forced segment"
    assert all(len(s.raw) <= 80 for s in got)


def test_segment_ids_are_ordered():
    seg = SpeechSegmenter()
    text = (
        "This is the first full sentence of the streamed answer. "
        "Here comes the second full sentence of the streamed answer. "
        "And finally the third full sentence arrives to complete it. "
    )
    got = feed_all(seg, text) + seg.finish()
    ids = [s.segment_id for s in got]
    assert ids == sorted(ids) == list(range(len(ids)))


def test_finish_flushes_remainder_and_marks_final():
    seg = SpeechSegmenter()
    feed_all(seg, "This trailing fragment has no closing punctuation at all")
    got = seg.finish()
    assert len(got) == 1
    assert got[0].final is True
    assert "trailing fragment" in got[0].text


def test_think_block_never_reaches_speech():
    seg = SpeechSegmenter()
    got = feed_all(
        seg,
        "<think>The user wants X so I should say Y.</think>"
        "Here is the actual spoken answer, which is long enough to emit. ",
    )
    got += seg.finish()
    joined = " ".join(s.text for s in got)
    assert "user wants X" not in joined
    assert "actual spoken answer" in joined


def test_think_tag_split_across_deltas():
    seg = SpeechSegmenter()
    out = []
    out += seg.feed("<thi")
    out += seg.feed("nk>hidden reasoning goes here</thi")
    out += seg.feed("nk>The visible answer text is certainly long enough to be emitted now. ")
    out += seg.finish()
    joined = " ".join(s.text for s in out)
    assert "hidden reasoning" not in joined
    assert "visible answer" in joined


def test_first_segment_breaks_early_at_clause():
    seg = SpeechSegmenter()
    # First segment: a clause boundary after ~16 chars is enough to start
    # speaking; the sentence continues in the next segment.
    got = feed_all(seg, "Sure, here is the plan, and then we will go through every step together. ")
    assert got, "first segment should emit at a clause boundary"
    assert got[0].raw.rstrip().endswith(",")
    assert len(got[0].raw) < 40


def test_later_segments_keep_full_thresholds():
    seg = SpeechSegmenter()
    feed_all(seg, "Sure, here is the plan for today. ")  # consumes segment 0
    # Now a short clause must NOT split; later segments wait for sentences.
    got = feed_all(seg, "First, do the thing. ")
    joined = " ".join(s.raw for s in got)
    assert "First, do the thing" in joined or got == []
    if got:
        assert not got[0].raw.rstrip().endswith(",")


def test_short_first_fragment_still_buffers():
    seg = SpeechSegmenter()
    assert feed_all(seg, "OK, ") == []  # a clause is not a sentence: buffer


def test_first_segment_emits_short_opening_clause():
    # Models often end the opener with a comma ("Sure,") despite the
    # prompt asking for a period; segment 0 speaks it either way.
    seg = SpeechSegmenter()
    got = feed_all(seg, "Sure, the capital of France is Paris and it has been ")
    assert got and got[0].segment_id == 0
    assert got[0].raw.strip() == "Sure,"


def test_first_segment_fast_path_respects_abbreviations():
    seg = SpeechSegmenter()
    got = feed_all(seg, "Dr. Smith can help with that question tomorrow morning. ")
    assert len(got) == 1
    assert "Dr. Smith" in got[0].raw  # "Dr." must not become the opener


def test_progressive_settle_across_segments():
    seg = SpeechSegmenter(min_words=8, min_chars=40, first_min_words=3, first_min_chars=12)
    # Segment 0: emits fast, clause break allowed, nothing left buffered.
    got = feed_all(seg, "Sure, okay then, ")
    assert [s.segment_id for s in got] == [0]
    assert len(got[0].raw) < 40  # broke well before the full threshold
    # Segment 1: midpoint thresholds — a 6-word, ~34-char sentence emits...
    got = feed_all(seg, "Now the second sentence ends here. ")
    assert [s.segment_id for s in got] == [1]
    # ...but segment 2+ needs the full 8 words / 40 chars, so the same
    # shape buffers instead.
    got = feed_all(seg, "And the third sentence ends now. ")
    assert got == []


def test_flush_stale_uses_clause_break():
    seg = SpeechSegmenter()
    # Consume segments 0 and 1 so this exercises the SETTLED thresholds
    # (segment 2+), where clause breaks are otherwise disallowed and
    # flush_stale is the only path to one.
    seg.feed("Here is the first full sentence of this answer, spoken now. ")
    seg.feed("And a second full sentence follows right after it, closing that. ")
    seg.feed("Well, considering everything you have described so far, and given the")
    got = seg.flush_stale()
    assert len(got) == 1
    assert got[0].raw.rstrip().endswith(",")


def test_second_segment_may_break_at_a_clause():
    # Segment 1 gates time-to-first-CONTENT: until it arrives the user has
    # heard only the acknowledgement and is sitting in silence. Measured
    # 2026-08-28: requiring a full sentence here cost ~1.7 s of dead air.
    seg = SpeechSegmenter()
    feed_all(seg, "Sure. ")  # consumes segment 0 (the acknowledgement)
    got = feed_all(
        seg,
        "The capital of France is Paris, and it has been the political "
        "centre of the country for many centuries. ",
    )
    assert got, "segment 1 should emit at the clause, not wait for the period"
    assert got[0].segment_id == 1
    assert got[0].raw.rstrip().endswith(",")


def test_third_segment_still_requires_a_sentence():
    # By segment 2 audio is flowing continuously, so prosody wins again.
    seg = SpeechSegmenter()
    feed_all(seg, "Sure. ")
    feed_all(seg, "The capital of France is Paris, and it has been for ages. ")
    got = feed_all(seg, "It sits on the river Seine, in the north of the country. ")
    if got:
        assert not got[0].raw.rstrip().endswith(","), (
            "later segments should not split at clauses"
        )
