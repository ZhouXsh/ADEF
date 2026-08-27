# ADEF 模型与训练优化变体

四组实验均由 0803 完整模型文件和完整训练脚本物理复制后直接修改得到。每个模型文件都包含 `DiffusionSchedule`、`DiTDecoderLayer`、`DiTDecoder`、`DenoisingNetwork` 与 `DitTalkingHead` 的完整实现，不通过导入、继承或 wrapper 复用样例实现。

## 1. Lip-aware residual + 嘴部专项监督

- 模型：`src/modules/emotion_dit_Unification_jianhua0803_lipaware.py`
- 训练：`train_Unification_jianhua0803_lipaware.py`
- 在共享 DiT 输出上增加零初始化的嘴部残差解码头，只修正六个嘴部隐式关键点对应的 18 个通道。
- 增加嘴部位置、速度、GT 加速度和窗口边界 Huber 损失。
- 情感分类损失不再向嘴部通道传梯度，减少情感张嘴与音素闭合之间的竞争。
- 这是最直接面向唇形同步的首选实验。

## 2. Multi-scale audio temporal pyramid + 分层微调

- 模型：`src/modules/emotion_dit_Unification_jianhua0803_audio_pyramid.py`
- 训练：`train_Unification_jianhua0803_audio_pyramid.py`
- 在 Wav2Vec2/HuBERT 投影后加入 3、5、9 帧的深度卷积时序金字塔，保持输出帧率和接口不变。
- 残差尺度零初始化，训练起点等价于原音频特征。
- 前期冻结完整音频编码器，随后只解冻顶部若干 Transformer 层，并使用较低的音频学习率。

## 3. Learnable emotion channel gate + 情感课程学习

- 模型：`src/modules/emotion_dit_Unification_jianhua0803_channelgate.py`
- 训练：`train_Unification_jianhua0803_channelgate.py`
- 保留“情感调制音频”核心逻辑，为每个音频特征通道学习 0 到 2 倍的调制强度，初始化时严格等价于原调制。
- 情感损失线性 warm-up，情感模块采用独立较小学习率。
- 同样阻断情感分类损失对嘴部通道的梯度，重点验证语音内容与情感表达的 Pareto 改善。

## 4. Stratified Min-SNR + AdamW + EMA

- 模型：`src/modules/emotion_dit_Unification_jianhua0803_minsnr_ema.py`
- 训练：`train_Unification_jianhua0803_minsnr_ema.py`
- 不改变网络主干，使用分层扩散时间步采样和适配 sample/noise target 的 Min-SNR 主损失。
- 优化器改为 AdamW，并维护 EMA 模型；checkpoint 的 `model` 保存 EMA 参数，`model_raw` 保存即时训练参数。
- 用于单独检验扩散优化稳定性，避免把结构收益与训练配方混在一起。

## 所有方案共有的修正

- `n_heads`、`n_layers`、`mlp_ratio`、`use_indicator` 与位置编码配置直接下传到 `DenoisingNetwork`。
- `sample()` 第三个返回值统一为未经情感调制的 `audio_feat_saved`。
- continuation 阶段将 prev 16 + current 64 波形合并后仅编码一次，再按帧切分。
- 冻结情感分类器参数；增加全局 seed 参数。
- 梯度累积时先除以 accumulation steps，只在真正 optimizer step 时裁剪梯度和更新 scheduler。

## 推荐顺序

1. Lip-aware；
2. Audio pyramid；
3. Channel gate；
4. Min-SNR + EMA。

先保持数据划分、batch size、训练步数、CFG 和随机种子一致，分别单独训练。不要一开始合并四组修改，否则无法判断指标提升来自哪里。

## 运行注意事项

- 四个训练脚本都会把 `args.model_module` 和 `args.optimization_variant` 写入 checkpoint，便于后续按对应的完整模型文件恢复。
- 四组模型均修正了 learnable positional encoding 的长度：位置参数与 `n_prev_motions + n_motions` 完全一致，不再多出一个未使用 token。
- 嘴部专项损失会依据 `end_idx` 屏蔽随机截断后的 padding 帧；速度和加速度损失使用相应的相邻有效帧掩码。
- 当 `gradient_accumulation_steps > 1` 时，scheduler 只在真实 `optimizer.step()` 后更新。
- Lip-aware、Audio pyramid 和 Channel gate 默认面向 `target=sample`；Min-SNR + EMA 同时兼容当前 `sample` 与 `noise` 目标，但建议先以 `sample` 做可比实验。

