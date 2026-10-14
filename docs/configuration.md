# Configuration

For direct host installs, the gateway reads environment variables and `.env`.
Compose sets gateway variables in `compose.yaml`. `.env` supplies values only
where the Compose file explicitly interpolates them.
[`.env.example`](../.env.example) lists the available settings and defaults.
This page explains how the main settings interact.

## Turn taking

`VAD_END_SILENCE_MS` (550) is how long someone has to stop talking before
the assistant confirms the end of a turn. Speculative generation can begin
earlier (`SPECULATIVE_SILENCE_MS`, 300); resumed speech cancels that
speculative response.
Lower end silence can cut off people who pause to think; higher values delay
playback.

`SEMANTIC_END_SILENCE_MS` (350) softens that trade. When the transcript
so far already reads as a complete question, the assistant stops waiting
sooner. Someone who says "where is the olive oil" gets the shorter
window. Someone who trails off mid-sentence gets the full one.

`UTTERANCE_MIN_RMS` (250) discards audio too quiet to be speech, which is
what keeps room noise from triggering a turn.

## Interruption

Speaking over the assistant cancels the rest of its reply.
`BARGE_IN_CONFIRM_MS` (250) is how much speech it takes to be believed,
and `BARGE_IN_MIN_RMS` (350) how loud. Both exist to avoid cancelling on
a cough.

The harder problem is the microphone hearing the assistant's own voice
when the browser's echo cancellation is imperfect. Without protection the assistant interrupts itself. `ECHO_SUPPRESS_SIMILARITY` (0.72)
compares what was just recognized against what was just spoken and drops
the turn if they match closely enough, within a window of
`ECHO_SUPPRESS_WINDOW_S` (15) seconds.

## Segmenting the answer

The model streams tokens. The segmenter decides when enough of them make
a speakable fragment. The first fragment is special.

`SEGMENT_FIRST_MIN_WORDS` (4) and `SEGMENT_FIRST_MIN_CHARS` (16) are
deliberately much lower than `SEGMENT_MIN_WORDS` (8) and
`SEGMENT_MIN_CHARS` (40). The first fragment is what ends the silence
after someone's question, so it is released at the first complete clause.
Later fragments wait for sentence boundaries, by which point audio is
continuous and prosody matters more than latency.

Raising the first-segment thresholds gives longer phrases but delays the
first speech. Measure that tradeoff on your host.

`SEGMENT_MAX_WAIT_MS` (1500) releases buffered text at the next model delta
when a boundary has not arrived. It does not impose a timeout on a stalled
model connection.

## The acknowledgment

`ACK_INJECT` plays a short prepared phrase ("Sure." / "Let me check.")
`ACK_AT_MS` (800) milliseconds after someone stops speaking, while the
answer is still being generated.

This cue does not reduce response latency. It signals that the question
was received while generation continues.

Below about 600 ms it steps on people who pause mid-sentence. Above about
1200 ms the silence before it is noticeable on its own.

`ACK_LONG_WHEN_BUSY` (2) switches to a longer phrase set when at least
that many answers are already generating, so more speech covers a wait that is about to be longer. Set it to 0 to always use the short set.

## Capacity

`MAX_ACTIVE_GENERATIONS` (4) is how many answers may generate at once.
On a CPU host the model is bound by memory bandwidth: adding
concurrent streams raises total throughput only until the memory channels
saturate, after which every stream, including the one that was already running, gets slower together. Past that point raising it makes
everybody wait.

`MAX_SESSIONS` (16 by default; 48 in `.env.example` and Compose) limits
concurrent connections. It also sizes
the streaming recognizer pool, one instance per possible speaker, so
raising it costs memory whether or not anyone connects.

Beyond `MAX_ACTIVE_GENERATIONS`, turns queue. A queue that never drains
indicates that the host is over capacity. Measure before adding generation
slots.

## Recognition

One recognizer: Moonshine v2 (`STT_MOONSHINE_V2_ARCH`, `medium-streaming`).
It decodes while the person is still talking, so end of speech costs
milliseconds rather than a full pass over the utterance, and it returns
punctuated, cased text, which the segmenter and the echo and endpointing
heuristics all depend on. The gateway warms one single-threaded instance
per admitted session (`MAX_SESSIONS`). A session speaks one utterance at
a time, so the pool is never outnumbered. An utterance the recognizer
returns nothing for does not start a turn.

The recognizer's workers need real cores. Pin the model server to a subset
of the physical cores and leave the rest to the services. Use Docker CPU
affinity or your process supervisor to assign cores, then measure latency
under load.

## Security

`AUTH_TOKEN` is a shared bearer token for session and voice endpoints, plus
`/api/ready` and `/api/metrics`. An empty token is allowed only when
`BIND_HOST` is loopback. Generate a long random value before binding to a
LAN address. The browser stores the token in local storage on that origin.

`ALLOWED_ORIGINS` is a JSON list. Empty means same-origin only.

`/api/health` returns only a basic status and remains unauthenticated for
health checks. Keep `/api/metrics` private at the reverse proxy as well.
