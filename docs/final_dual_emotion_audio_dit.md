# Final dual-branch emotion-audio DiT variants

This branch adds three copy-style DiT variants.  The original files are not modified.

## Core design

All three variants use the same DiT condition structure:

```text
motion query
  ├── cross-attend original audio memory         -> lip-sync / speech content
  └── cross-attend emotion-modulated audio memory -> emotional expression
```

The key difference among the variants is how the emotion-modulated audio feature is built.

## `src/modules/emotion_dit_finalv1.py`

Based on `emotion_dit_clean.py` and the original `emotion_dit.py` AdaLN emotion modulation.

```text
emotion label -> embedding -> shift/scale
A_e = LN(A) * (1 + scale) + shift
```

This version is the most direct comparison with the original `emotion_dit.py`, except that it keeps the original audio branch and the emotion-modulated audio branch separated in DiT cross-attention.

## `src/modules/emotion_dit_finalv2.py`

Based on `emotion_dit_clean_encoding.py`.

```text
emotion label -> [B, K, D] emotion basis bank
A queries the emotion basis bank
A_e = A + alpha * Gate(A, Attn(A, P_y))
```

This version treats the emotion library as an audio recalibration module rather than an independent DiT condition.

## `src/modules/emotion_dit_finalv3.py`

Based on the clean dual-branch idea and the emotion2vec path.

```text
label y -> target emotion prototype bank P_y
utterance emotion2vec u -> globally calibrates P_y
frame emotion2vec F -> redirected by calibrated P_y^u
A queries [P_y^u, u, F_y]
A_e = A + alpha * gated residual
```

This version implements the hierarchical emotion encoder:

1. label defines target emotion direction;
2. utterance emotion2vec calibrates global intensity/style;
3. frame emotion2vec provides local affect dynamics;
4. the three emotion features act on audio to produce emotion-fused audio.

## CFG

All three variants use clean three-branch CFG when both `audio` and `emotion` are enabled:

```text
branch 0: null
branch 1: audio-only
branch 2: audio + emotion-audio

prediction = null
           + s_audio * (audio_only - null)
           + s_emotion * (audio_emotion - audio_only)
```

This preserves the method story:

- audio guidance controls speech-content/lip-sync;
- emotion-audio guidance adds emotional expression on top of audio.

## Notes

- `finalv1` and `finalv2` keep the original training call signature.
- `finalv3` accepts optional `emo_utt_feat`, `emo_frame_feat`, and `prev_emo_frame_feat` in `forward()` and `sample()`.
- If emotion2vec features are omitted in `finalv3`, learned null utterance/frame tokens are used, so the model still runs as a label-conditioned variant.
