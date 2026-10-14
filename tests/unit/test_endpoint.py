from gateway.orchestration.endpoint import looks_complete


def test_complete_utterances():
    assert looks_complete("What is the capital of France?")
    assert looks_complete("Tell me about the Simpsons.")
    assert looks_complete("turn off the kitchen lights")  # unpunctuated ASR
    assert looks_complete("what's the weather tomorrow")


def test_incomplete_utterances():
    assert not looks_complete("What is the capital of")
    assert not looks_complete("I want to")
    assert not looks_complete("tell me about the")
    assert not looks_complete("So I was thinking that")
    assert not looks_complete("Can you please, um")
    assert not looks_complete("and then he said,")


def test_conservative_edges():
    assert not looks_complete("")
    assert not looks_complete("yes")  # single word: wait for full window
    assert not looks_complete("okay so")
