# emotion2vec-conditioned ADEF training copy

This PR adds a copy-style emotion2vec training path without modifying the original ADEF files.

## New files

- `src/modules/emotion_dit_e2v.py`
  - Adds a DiT backbone with decoupled condition attention.
  - Uses HuBERT/Wav2Vec audio memory for content and lip-sync.
  - Uses emotion-label basis tokens as discrete affect anchors.
  - Uses emotion2vec utterance features as global affect/prosody tone.
  - Uses emotion2vec frame features as local temporal affect memory.

- `src/dataset/dataset_EmotionLevel_e2v.py`
  - Copy of the original MEAD dataset reader with additional emotion2vec feature loading.
  - Expected layout:

```text
videos/<speaker>/front/<emotion>/<level>/
    <name>.wav
    frame/<name>.npy
    utterance/<name>.npy
```

- `src/utils/e2v_losses.py`
  - Adds emotion2vec-motion prosody curve consistency loss.
  - Compares the temporal trend of audio affect intensity and visual motion affect intensity.

- `train_e2v.py`
  - Copy-style training entrypoint for the new model and dataset.

## Example command

```bash
python train_e2v.py \
  --data_root src/my_prepare/ \
  --motion_filename front_all_motions.pkl \
  --motion_template_filename motion_template.pkl \
  --audio_model wav2vec2 \
  --e2v_dim 1024 \
  --l_emo_cls 1.0 \
  --l_emo_level 0.2 \
  --l_prosody_curve 0.02 \
  --batch_size 16 \
  --device_id 0
```

If the paths in `train.txt` point directly to the MEAD audio/video files, the dataset will find emotion2vec features under `level_X/frame/` and `level_X/utterance/` automatically.  If the features are stored under a different root, pass `--emotion2vec_root`.

## Loss interpretation

The existing emotion classifier loss answers: "does the generated motion match the target emotion category?"

The new prosody curve loss answers: "does the generated expression intensity evolve over time like the emotion2vec frame-level audio affect curve?"

This keeps the category control and temporal affect dynamics separate.
