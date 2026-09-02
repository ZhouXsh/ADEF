# Pre-trained Evaluation Models

This directory hosts the pre-trained weights used by the unified evaluation pipeline.

| File | Size | Source | Used for |
| --- | --- | --- | --- |
| `syncnet_v2.model` | ~52 MB | https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model (mirror of the original VGG/Joon Son Chung release) | Audio-lip sync (LSE-C / SyncConf) |
| `shape_predictor_68_face_landmarks.dat` | ~95 MB | http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 | LMD / mouth landmark distance |
| `Resnet18_FER+_pytorch.pth.tar` | ~43 MB | Baidu/OneDrive (see `../Emotion-FAN/pretrain_model/readme.md`) — download and place here | Emotion-FAN classifier for Accemo |

## Manual download steps

The two heavy weights are not auto-fetched because they sit behind third-party cloud
storage that does not allow plain HTTP downloads from this environment:

1. **SyncNet v2** — already downloaded at `syncnet_v2.model`.
2. **dlib 68 landmarks** — already downloaded at `shape_predictor_68_face_landmarks.dat`.
3. **Emotion-FAN ResNet18** — fetch from the link inside
   `../Emotion-FAN/pretrain_model/readme.md` (Baidu Netdisk or OneDrive), then drop
   the file here as `Resnet18_FER+_pytorch.pth.tar`.