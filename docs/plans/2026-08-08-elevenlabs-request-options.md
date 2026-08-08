# ElevenLabs Request Options Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Make Hermes forward configured ElevenLabs speed and text-normalization options in normal and streaming TTS requests.

**Architecture:** Add one shared option builder in `tools/tts_tool.py`. It validates `tts.elevenlabs.apply_text_normalization`, builds an SDK `VoiceSettings` object only when speed is configured, and is reused by the normal and streaming providers.

**Tech Stack:** Python 3.11, ElevenLabs SDK 1.59.0, pytest, Ruff.

---

### Task 1: Normal ElevenLabs requests

**Files:**
- Modify: `tools/tts_tool.py`
- Create: `tests/tools/test_tts_elevenlabs_options.py`

1. Write a failing test asserting `apply_text_normalization="auto"` and `VoiceSettings(speed=1.2)` reach `text_to_speech.convert()`.
2. Run the focused test and confirm the request omits both options before implementation.
3. Add the minimal shared option builder and use it in `_generate_elevenlabs()`.
4. Run the focused test green.

### Task 2: Streaming requests

**Files:**
- Modify: `tools/tts_streaming.py`
- Modify: `tests/tools/test_tts_elevenlabs_options.py`

1. Write a failing streaming test for the same options.
2. Run it red.
3. Reuse the shared option builder in `ElevenLabsStreamer.stream()`.
4. Run both tests green.

### Task 3: Validation and documentation

**Files:**
- Modify: `tests/tools/test_tts_elevenlabs_options.py`
- Modify: `website/docs/user-guide/features/tts.md`

1. Add a failing test rejecting unsupported normalization values.
2. Add minimal validation for `auto`, `on`, and `off`.
3. Document `tts.elevenlabs.speed` and `apply_text_normalization`.
4. Run focused TTS tests, Ruff, compile, and `git diff --check`.

### Task 4: Delivery and runtime

1. Commit and push the verified branch.
2. Update the detached runtime candidate to the verified commit and sync `dev`, `messaging`, `hindsight`, and `tts-premium` extras.
3. Set config through `hermes config set`.
4. Generate real ElevenLabs audio and verify a non-empty audio artifact.
5. Proceed to the separately requested n8n skills audit and installation before the final owner restart.
