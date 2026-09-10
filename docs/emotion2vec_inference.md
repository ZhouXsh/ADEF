# Emotion2vec-conditioned inference

This inference path is intended for the following motion generators:

- `src/modules/emotion_dit_e2v.py`
- `src/modules/emotion_dit_finalv3.py`
- `src/modules/emotion_dit_finalv3_two_stage.py`

It is isolated from the legacy `inference.py` entrypoint. Use
`inference_e2v.py` for these three model families.

## Runtime feature flow

For every input audio file, the pipeline obtains two conditions using the same
FunASR emotion2vec method as `src/my_prepare/06_extract_emotion2vec.py`:

1. utterance-level embedding: `[1, 1024]`;
2. frame-level embedding: `[1, T, 1024]`.

The frame feature is interpolated to the ADEF video-frame timeline, aligned
with the wrapper's fixed audio padding, divided into `n_motions` windows, and
passed to `sample()` as:

- `emo_utt_feat`;
- `emo_frame_feat`;
- `prev_emo_frame_feat` for the preceding temporal context.

The original HuBERT/Wav2Vec input remains unchanged and is still responsible
for speech content and lip synchronization.

## Basic command

```bash
python inference_e2v.py \
  --reference assets/examples/imgs/joyvasa_001.png \
  --audio assets/examples/audios/joyvasa_001.wav \
  --checkpoint-MotionGenerator /path/to/checkpoint.pt \
  --motion-template-path /path/to/motion_template.pkl \
  --motion-generator-variant auto \
  --emotype happy \
  --output-dir new_animations
```

`--motion-generator-variant auto` inspects checkpoint arguments and parameter
names. It can also be set explicitly to one of:

```text
emotion_dit_e2v
emotion_dit_finalv3
emotion_dit_finalv3_two_stage
```

For a two-stage final-v3 checkpoint, the loader switches the model to Stage 2
before inference so that the hierarchical emotion branch is active.

## Feature cache

By default, generated emotion2vec features are stored next to the audio:

```text
<the audio directory>/.adef_emotion2vec/
├── utterance/<audio stem>.npy
└── frame/<audio stem>.npy
```

A separate cache root can be selected with:

```bash
--emotion2vec-cache-dir /path/to/cache
```

To regenerate both embeddings:

```bash
--emotion2vec-force-extract True
```

## Use precomputed features

Both paths must be provided together:

```bash
python inference_e2v.py \
  ... \
  --emotion2vec-utterance-path /path/to/utterance.npy \
  --emotion2vec-frame-path /path/to/frame.npy
```

The accepted shapes are:

- utterance: `[D]`, `[1, D]`, or a sequence that can be averaged to `[D]`;
- frame: `[T, D]` or `[1, T, D]`;
- `D` must match the model's `e2v_dim`, normally 1024.

If explicit files are supplied, FunASR is not initialized.

## CFG

The three supported model families use incremental condition guidance:

```text
null -> audio only -> audio + emotion
```

Typical configuration:

```bash
--cfg-scale 2.8
```

For separate audio and emotion strengths, pass a list according to the Tyro
CLI syntax used by the local environment. The checkpoint's default guiding
conditions are used when `cfg_cond` is omitted.

## Motion cache safety

The old `save_results` cache uses only the audio filename for its `.pkl` path.
That filename does not encode the target emotion, model checkpoint, or
emotion2vec feature files. Therefore `inference_e2v.py` disables that legacy
motion cache by default even if `--save-results True` is supplied.

It can be re-enabled deliberately with:

```bash
--allow-legacy-motion-cache True
```

This is only safe when the target emotion, checkpoint, CFG settings, and e2v
features are guaranteed not to change.

## Dependencies

The repository already lists `funasr` and `modelscope` in `requirements.txt`.
The first online extraction may download `iic/emotion2vec_plus_large`. For an
offline machine, precompute the two `.npy` files and provide their paths.
