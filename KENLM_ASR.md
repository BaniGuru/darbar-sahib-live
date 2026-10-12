# Gurbani KenLM streaming decoding

The streaming RNNT supports the supplied 3-, 4- and 5-gram models; `kirtan_stream_model/gurbani_3gram.bin` is selected alongside the existing soft Gurbani corpus prior. The binary is stored in Git LFS. The acoustic encoder, predictor, tokenizer and joint weights are unchanged.

## Install and run

Use the existing checkout at `/workspace/darbar-sahib-live`. On Linux, KenLM 0.3.0 needs a C++ compiler when a compatible wheel is unavailable. The current environment was built and tested with:

```bash
git lfs pull
CC=gcc CXX=g++ .venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python -m unittest discover -s tests
FFMPEG_PATH=/usr/bin/ffmpeg PORT=8000 .venv/bin/uvicorn server:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

The full application's `/asr/health` reports the active KenLM model, order, weight, scoring mode, word bonus and beam width. The standalone ASR application is `server_asr_stream:app`; its corresponding endpoint is `/health`.

## Scoring and cache boundaries

The selected defaults are 24 beams, KenLM weight 0.3, `ngram` mode and zero word bonus. The existing corpus prior keeps weight 0.6.

KenLM scores complete words during beam expansion. Its native log10 probabilities are converted to natural logarithms before combining them with acoustic scores. Probability mode uses a per-word bonus to balance the model's tendency to prefer shorter transcripts; word costs are capped at 12 nats so unknown commentary remains possible. Ratio mode compares contextual probability with the unigram; ngram mode limits that evidence to observed matches of order three or higher; full_order mode requires the model's maximum order. These softer modes cap contextual evidence and allow unknown words.

The LM keeps order minus one completed words of context (two, three or four) across encoder cache promotions. An unfinished word remains in the acoustic beam until its boundary resolves. EOF scores the terminal word without adding audio or an artificial sentence boundary. The existing 800 ms partial refresh, four-second acoustic cache steps, one-second word lookahead, and eight-second maximum pending window remain in effect.

Native states, word scores and completed token prefixes use bounded caches. KenLM and the corpus prior share token-boundary decoding, avoiding repeated SentencePiece work on every preview. The 16,384-entry RNNT predictor cache continues to reuse exact predictor computations.

## Controls and validation

`ASR_KENLM_WEIGHT=0` disables KenLM and retains the existing streaming decoder. Advanced tuning uses `ASR_KENLM_PATH`, `ASR_KENLM_MODE` (`ngram`, `full_order`, `probability` or `ratio`; `trigram` is a legacy alias for `ngram`), `ASR_KENLM_WORD_BONUS`, `ASR_LANGUAGE_PRIOR_WEIGHT` and `ASR_BEAM_WIDTH`. Specify the weight and bonus together when changing scoring modes.

The [section-24 comparison](validation/kenlm-section24/README.md) compares all three orders and selects the defaults. The [earlier 20-minute comparison](validation/kenlm20m/README.md) records 3-gram profiles, whole-recording WER/CER, missing words, rejected settings and speed measurements. It includes complete transcripts and a highlighted word diff. Only terminal onkar ੁ and sihari ਿ are ignored in the primary accuracy comparison; internal vowels, repetitions and word boundaries remain significant.

```bash
.venv/bin/python scripts/evaluate_reference_asr.py \
  --audio /workspace/offline-model-validation/first20m.wav \
  --reference /workspace/offline-model-validation/reference20m.json \
  --ignore-final-short-vowels --report /tmp/kenlm20m.json
```

The recognizer receives audio only, without reference text or timestamps. The supplied binary's training corpus was not provided; these results validate this recording rather than establish independent held-out accuracy.
