<h1 align='center'>ADEF: Research on Audio-Driven
Emotion-Fused Talking Video Generation</h1>


## 📖 Introduction

## 🧳 Framework

![Inference Pipeline](assets/imgs/pipeline.png)

## ⚙️ Installation

**System requirements:**

Ubuntu:

- Tested on Ubuntu 20.04, CUDA 12.1
- Tested GPUs: RTX 4090

Windows:

- Tested on Windows 11, CUDA 12.1
- Tested GPUs: RTX 4060 Laptop 8GB VRAM GPU

**Create environment:**

```bash
# 1. Create base environment
conda create -n adef python=3.10.16 -y
conda activate adef 

# 2. Install requirements
pip install -r requirements.txt

# 3. Install ffmpeg
sudo apt-get update  
sudo apt-get install ffmpeg -y

# 4. Optional: Install MultiScaleDeformableAttention for animal image animation
cd src/utils/dependencies/XPose/models/UniPose/ops
python setup.py build install
cd - # equal to cd ../../../../../../../
```

## 🎒 Prepare model checkpoints

Make sure you have [git-lfs](https://git-lfs.com) installed and download all the following checkpoints to `pretrained_weights`:

### 1. Download ADEF checkpoints

```bash
git lfs install
git clone XXXX
```

### 2. Download audio encoder checkpoints

We suport two types of audio encoders, including [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), and [hubert-chinese](https://huggingface.co/TencentGameMate/chinese-hubert-base).

Run the following commands to download [hubert-chinese](https://huggingface.co/TencentGameMate/chinese-hubert-base) pretrained weights:

```bash
git lfs install
git clone https://huggingface.co/TencentGameMate/chinese-hubert-base
```

To get the [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h) pretrained weights, run the following commands:

```bash
git lfs install
git clone https://huggingface.co/facebook/wav2vec2-base-960h
```

### 3. Download LivePortraits checkpoints

```bash
# !pip install -U "huggingface_hub[cli]"
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

Refering to [Liveportrait](https://github.com/KwaiVGI/LivePortrait/tree/main) for more download methods.

### 4. `pretrained_weights` contents

The final `pretrained_weights` directory should look like this:

```text
./pretrained_weights/
├── ADEF
│   ├── audio2emo
│   │   └── audio2emo.pth
│   ├── emo_classifier
│   │   └── emo_level_classifier.pth
│   ├── emo_enhancer
│   │   └── emo_enhancer.pth
│   ├── motion_generator
│   │   └── motion_generator.pt
│   └── motion_template
│       └── motion_template.pkl
├── insightface                                                                                                                                                 
│   └── models                                                                                                                                                  
│       └── buffalo_l                                                                                                                                           
│           ├── 2d106det.onnx                                                                                                                                   
│           └── det_10g.onnx   
├── liveportrait
│   ├── base_models
│   │   ├── appearance_feature_extractor.pth
│   │   ├── motion_extractor.pth
│   │   ├── spade_generator.pth
│   │   └── warping_module.pth
│   ├── landmark.onnx
│   └── retargeting_models
│       └── stitching_retargeting_module.pth
├── TencentGameMate:chinese-hubert-base
│   ├── chinese-hubert-base-fairseq-ckpt.pt
│   ├── config.json
│   ├── gitattributes
│   ├── preprocessor_config.json
│   ├── pytorch_model.bin
│   └── README.md
└── wav2vec2-base-960h               
    ├── config.json                  
    ├── feature_extractor_config.json
    ├── model.safetensors
    ├── preprocessor_config.json
    ├── pytorch_model.bin
    ├── README.md
    ├── special_tokens_map.json
    ├── tf_model.h5
    ├── tokenizer_config.json
    └── vocab.json
```

> [!NOTE]
> The folder `TencentGameMate:chinese-hubert-base` in Windows should be renamed `chinese-hubert-base`.

## 🚀 Inference

### 1. Inference with command line

```python
python inference.py -r assets/examples/imgs/XXX.png -a assets/examples/audios/XXX.wav --cfg_scale 2.0
```

You can change cfg_scale to get results with different expressions and poses.

### 2. Inference with web demo

Use the following command to start web demo:

```python
python app.py
```

The demo will be create at http://127.0.0.1:7862.


## ⚓️ Train Motion Generator with Your Own Data

The motion generater should be trained using human talking face videos.


### 1. Prepare train and validation data

Change the `root_dir` in `01_extract_motions.py` with you own dataset path, then run the following commands to generate training and validation data:

```bash
cd src/prepare_data
python 01_extract_motions.py
python 02_gen_labels.py
pyhton 03_merge_motions.py
python 04_gen_template.py

mv motion_templete.pkl motions.pkl train.json test.json ../../data
cd ../..
```

### 2. Train

```bash
python train.py
```

The experimental results is located in `experiments/`.

## 🤝 Acknowledgments

We would like to thank the contributors to the 
[JoyVASA](https://github.com/KwaiVGI/LivePortrait),[LivePortrait](https://github.com/KwaiVGI/LivePortrait), [Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [InsightFace](https://github.com/deepinsight/insightface), [X-Pose](https://github.com/IDEA-Research/X-Pose), [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk), [Hallo](https://github.com/fudan-generative-vision/hallo), [wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec), [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain), [Q-Align](https://github.com/Q-Future/Q-Align), [Syncnet](https://github.com/joonson/syncnet_python), and [VBench](https://github.com/Vchitect/VBench) repositories, for their open research and extraordinary work.
