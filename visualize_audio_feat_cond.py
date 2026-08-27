# -*- coding: utf-8 -*-
"""可视化主干 DitTalkingHead 的 audio_feat_cond。

读取 ``experiments/emo_dit/20260815_fuke_20260806_emotion_dit_Unification_bs128_lexp1/checkpoints/iter_0300000.pt``
并以 ``src/my_prepare/train.txt`` 列出的训练集音频为输入，复现 emotion_dit_Unification_jianhua0803.py
中 ``forward`` 路径下的情感调制：

    emo_feat     = self.emo_embed(emo_index).unsqueeze(1)
    emo_shift,
    emo_scale    = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
    audio_feat_cond = self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

对得到的 64 帧 × 512 维特征沿时间维取平均，再用 t-SNE / UMAP 降至 2D 并按情感着色。
同一份特征也保存成 .npz，便于复用。

Usage:
    /home/Zhouxishi/miniconda3/envs/ADEFv4/bin/python visualize_audio_feat_cond.py \
        --checkpoint experiments/emo_dit/20260815_fuke_20260806_emotion_dit_Unification_bs128_lexp1/checkpoints/iter_0300000.pt \
        --data_root src/my_prepare \
        --n_per_emo 30 \
        --out_dir experiments/emo_dit/20260815_fuke_20260806_emotion_dit_Unification_bs128_lexp1/visualizations
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data

# 把 ADEF_remake 加入 sys.path，这样可以直接 import src.* 与 emotion_dit_* ；
# 必须在 import 任何 src.* 模块之前完成。
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 训练时使用的情感列表是旧版（含 'contempt'），当前 src/config/emotion_config.py 已经把
# 'contempt' 改名为 'calm'。这里在 import config 之前 monkey-patch 回来，避免 dataset
# 报 ``'contempt' is not in list``。训练时 checkpoint 的 emo_embed 形状也是 8，对应这一顺序。
import src.config.emotion_config as _emo_cfg
_EMO_LIST_ORIG = ['angry', 'contempt', 'disgusted', 'fear',
                  'happy', 'neutral', 'sad', 'surprised']
_emo_cfg.global_emo_list = _EMO_LIST_ORIG  # type: ignore[attr-defined]

from src.modules.emotion_dit_Unification_jianhua0803 import DitTalkingHead  # noqa: E402
from src.dataset.dataset_EmotionLevel_clear_jianhua0803 import EmoLevelDataset  # noqa: E402

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


# ---------------------------------------------------------------------------
# 1. 模型加载
# ---------------------------------------------------------------------------
def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> tuple[DitTalkingHead, dict]:
    """从训练 checkpoint 重建 DitTalkingHead 并加载权重。

    模型结构参数从 checkpoint 中的 ``args`` 读取，避开手工指定默认值带来的潜在偏差。
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']

    model_kwargs = dict(
        device=device,
        target=args.target,
        architecture=args.architecture,
        motion_feat_dim=args.motion_feat_dim,
        fps=args.fps,
        n_motions=args.n_motions,
        n_prev_motions=args.n_prev_motions,
        audio_model=args.audio_model,
        feature_dim=args.feature_dim,
        n_diff_steps=args.n_diff_steps,
        diff_schedule=args.diff_schedule,
        cfg_mode=args.cfg_mode,
        guiding_conditions=args.guiding_conditions,
        emo_classes=len(_EMO_LIST_ORIG),
        align_mask_width=args.align_mask_width,
    )

    model = DitTalkingHead(**model_kwargs)

    # 关键：若 cls token / 不匹配键，则只加载能对齐的部分。
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        # 这里仅打印；非致命，例如 dataset wrapper 多包一层时会出现 unexpected。
        print(f'[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}')
        if missing[:5]:
            print('  first missing  :', missing[:5])
        if unexpected[:5]:
            print('  first unexpected:', unexpected[:5])

    model.eval()
    model.to(device)
    return model, ckpt


# ---------------------------------------------------------------------------
# 2. 训练集均衡采样
# ---------------------------------------------------------------------------
class BalancedSubset(data.Dataset):
    """对底层 EmoLevelDataset 按情感类别均衡采样 n_per_emo 条样本。

    原始 train.txt 中 8 类共 ~30000 条；这里按情感聚合索引，再对每个情感做有放回 / 无放回
    抽样。同一视频不同 frame 切片也算一条独立样本，所以这里允许重复切片（无放回）。
    """

    def __init__(self, base: EmoLevelDataset, n_per_emo: int, seed: int = 0):
        self.base = base
        self.n_per_emo = n_per_emo
        self.rng = np.random.default_rng(seed)

        # 按情感聚合全局索引
        by_emo: dict[int, list[int]] = {i: [] for i in range(len(_EMO_LIST_ORIG))}
        for idx, item in enumerate(base.all_data):
            emotype = item['video_name'].split('/')[-1].split('_')[2]
            if emotype in _EMO_LIST_ORIG:
                by_emo[_EMO_LIST_ORIG.index(emotype)].append(idx)

        self.indices: list[int] = []
        self.emo_indices: list[int] = []
        for emo_id, idxs in by_emo.items():
            if not idxs:
                print(f'[WARN] emotion {emo_id} ({_EMO_LIST_ORIG[emo_id]}) has 0 sample')
                continue
            if len(idxs) >= n_per_emo:
                chosen = self.rng.choice(idxs, size=n_per_emo, replace=False)
            else:
                chosen = self.rng.choice(idxs, size=n_per_emo, replace=True)
            self.indices.extend(chosen.tolist())
            self.emo_indices.extend([emo_id] * n_per_emo)

        # 重新打乱
        perm = self.rng.permutation(len(self.indices))
        self.indices = [self.indices[i] for i in perm]
        self.emo_indices = [self.emo_indices[i] for i in perm]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        # 复写以确保返回我们记录的 emo_id（与 audio_path 解析一致）；底层 dataset 会再解一次，
        # 但内容应该匹配。
        audio, coef, emo_idx, emo_lvl = self.base[int(self.indices[i])]
        return audio, coef, torch.as_tensor(self.emo_indices[i]), emo_lvl


# ---------------------------------------------------------------------------
# 3. 特征提取
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_audio_feat_cond(model: DitTalkingHead,
                            loader: data.DataLoader,
                            device: torch.device,
                            audio_unit: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """遍历 loader，提取每条样本的 audio_feat_cond。

    返回：
        feats_cond : (N, n_motions, feature_dim) 调制后的音频特征
        feats_norm : (N, n_motions, feature_dim) 仅 LayerNorm、未注入情感的参考特征
        labels     : (N,) 对应情感索引
    """
    all_cond, all_norm, all_lbl = [], [], []
    n_done = 0
    t0 = time.time()
    for audio, _coef, emo_index, _level in loader:
        audio = audio.to(device)               # (B, audio_total_len)
        emo_index = emo_index.to(device)       # (B,)

        # 与训练一致：只取最后 n_motions 帧的音频（"current" 段）
        n_audio_samples = round(audio_unit * model.n_motions)
        current_audio = audio[:, -n_audio_samples:].contiguous()  # (B, n_motions * audio_unit)

        # 1) 原始音频特征
        audio_feat = model.extract_audio_feature(current_audio)   # (B, n_motions, feature_dim)
        audio_feat = audio_feat.contiguous()

        # 2) 仅 LayerNorm
        audio_norm = model.audio_norm(audio_feat)                # (B, n_motions, feature_dim)

        # 3) 情感调制：与 forward 中 audio_feat_cond 完全一致
        emo_feat = model.emo_embed(emo_index).unsqueeze(1)       # (B, 1, feature_dim)
        emo_shift, emo_scale = model.adaLN_modulation(emo_feat).chunk(2, dim=2)
        audio_cond = audio_norm * (1 + emo_scale) + emo_shift    # (B, n_motions, feature_dim)

        all_cond.append(audio_cond.cpu().float().numpy())
        all_norm.append(audio_norm.cpu().float().numpy())
        all_lbl.append(emo_index.cpu().numpy())
        n_done += audio.shape[0]
        if n_done % (audio.shape[0] * 5) == 0 or n_done == audio.shape[0]:
            print(f'  ... extracted {n_done} samples  (elapsed {time.time()-t0:.1f}s)')

    feats_cond = np.concatenate(all_cond, axis=0)
    feats_norm = np.concatenate(all_norm, axis=0)
    labels = np.concatenate(all_lbl, axis=0).astype(np.int64)
    return feats_cond, feats_norm, labels


# ---------------------------------------------------------------------------
# 4. 降维 & 画图
# ---------------------------------------------------------------------------
def reduce_and_plot(feats_cond: np.ndarray,
                    feats_norm: np.ndarray,
                    labels: np.ndarray,
                    out_dir: Path,
                    perplexity: int,
                    umap_neighbors: int,
                    seed: int) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE
    import umap

    # 取时间维均值 -> (N, feature_dim)
    mean_cond = feats_cond.mean(axis=1)
    mean_norm = feats_norm.mean(axis=1)

    # 标准化（按样本 L2 归一化），让聚类更看重方向
    def _l2(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return x / n

    Xc = _l2(mean_cond)
    Xn = _l2(mean_norm)

    print(f'[reduce] feats shape cond={Xc.shape} norm={Xn.shape}')

    # ---- UMAP ----
    print('[reduce] fitting UMAP (cond) ...')
    reducer_c = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.3,
                          metric='cosine', random_state=seed, verbose=False)
    emb_c_umap = reducer_c.fit_transform(Xc)

    print('[reduce] fitting UMAP (norm) ...')
    reducer_n = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.3,
                          metric='cosine', random_state=seed, verbose=False)
    emb_n_umap = reducer_n.fit_transform(Xn)

    # ---- t-SNE ----
    print('[reduce] fitting t-SNE (cond) ...')
    perp = min(perplexity, max(5, (len(Xc) - 1) // 3))
    tsne_c = TSNE(n_components=2, perplexity=perp, init='pca',
                  learning_rate='auto', random_state=seed, max_iter=1500)
    emb_c_tsne = tsne_c.fit_transform(Xc)

    print('[reduce] fitting t-SNE (norm) ...')
    tsne_n = TSNE(n_components=2, perplexity=perp, init='pca',
                  learning_rate='auto', random_state=seed, max_iter=1500)
    emb_n_tsne = tsne_n.fit_transform(Xn)

    # ---- 画图 ----
    # 8 类情感，用 dataviz palette 顺序的 categorical slots（已验证相邻色对，CVD 安全）
    palette = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
               '#e87ba4', '#008300', '#4a3aa7', '#e34948']
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    surface = '#fcfcfb'
    ink_primary = '#0b0b0b'
    ink_secondary = '#52514e'
    hairline = '#e1e0d9'

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=140)
    fig.patch.set_facecolor(surface)

    panels = [
        (axes[0, 0], emb_c_umap, 'UMAP · audio_feat_cond (post emotion modulation)'),
        (axes[0, 1], emb_n_umap, 'UMAP · audio_norm(audio_feat) (pre emotion modulation)'),
        (axes[1, 0], emb_c_tsne, 't-SNE · audio_feat_cond (post emotion modulation)'),
        (axes[1, 1], emb_n_tsne, 't-SNE · audio_norm(audio_feat) (pre emotion modulation)'),
    ]

    for ax, emb, title in panels:
        ax.set_facecolor(surface)
        for emo_id in range(len(_EMO_LIST_ORIG)):
            mask = labels == emo_id
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=palette[emo_id], marker=markers[emo_id],
                       s=46, linewidths=0.6, edgecolor='white',
                       alpha=0.85, label=_EMO_LIST_ORIG[emo_id])
        ax.set_title(title, color=ink_primary, fontsize=11.5, pad=8)
        ax.set_xlabel('dim-1', color=ink_secondary, fontsize=9)
        ax.set_ylabel('dim-2', color=ink_secondary, fontsize=9)
        ax.tick_params(colors=ink_secondary, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(hairline)
        ax.grid(True, color=hairline, linewidth=0.6, alpha=0.7)

    # 自定义图例（散点 + 标签）
    handles = [Line2D([0], [0],
                      marker=markers[i], linestyle='',
                      markerfacecolor=palette[i], markeredgecolor='white',
                      markersize=8, label=_EMO_LIST_ORIG[i])
               for i in range(len(_EMO_LIST_ORIG))]
    leg = fig.legend(handles=handles,
                     loc='lower center', ncol=4, frameon=False,
                     bbox_to_anchor=(0.5, -0.005),
                     labelcolor=ink_primary, fontsize=10, handletextpad=0.5,
                     columnspacing=1.4)

    fig.suptitle('DitTalkingHead · audio_feat_cond visualization (trainset, 8 emotions)',
                 color=ink_primary, fontsize=13, y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig_path = out_dir / 'audio_feat_cond_2x2.png'
    fig.savefig(fig_path, dpi=180, bbox_inches='tight', facecolor=surface)
    plt.close(fig)
    print(f'[save] figure -> {fig_path}')

    # 同时存一个单独的 cond 主图（UMAP），方便查看
    fig2, ax2 = plt.subplots(figsize=(8, 7), dpi=140)
    fig2.patch.set_facecolor(surface)
    ax2.set_facecolor(surface)
    for emo_id in range(len(_EMO_LIST_ORIG)):
        mask = labels == emo_id
        ax2.scatter(emb_c_umap[mask, 0], emb_c_umap[mask, 1],
                    c=palette[emo_id], marker=markers[emo_id],
                    s=56, linewidths=0.6, edgecolor='white',
                    alpha=0.9, label=_EMO_LIST_ORIG[emo_id])
    ax2.set_title('audio_feat_cond (UMAP) · same-emotion samples should cluster together',
                  color=ink_primary, fontsize=12, pad=10)
    ax2.set_xlabel('UMAP-1', color=ink_secondary, fontsize=10)
    ax2.set_ylabel('UMAP-2', color=ink_secondary, fontsize=10)
    ax2.tick_params(colors=ink_secondary, labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color(hairline)
    ax2.grid(True, color=hairline, linewidth=0.6, alpha=0.7)
    ax2.legend(loc='best', frameon=False, labelcolor=ink_primary, fontsize=10)
    single_path = out_dir / 'audio_feat_cond_umap.png'
    fig2.tight_layout()
    fig2.savefig(single_path, dpi=180, bbox_inches='tight', facecolor=surface)
    plt.close(fig2)
    print(f'[save] figure -> {single_path}')

    # ---- 量化：类内 / 类间 平均余弦距离 ----
    def _intra_inter(X: np.ndarray, y: np.ndarray) -> tuple[float, float, dict]:
        n = X.shape[0]
        idxs = np.arange(n)
        # 同一对只能统计一次：i < j
        same, diff, same_per_emo, diff_per_emo = [], [], {i: [] for i in range(8)}, {i: [] for i in range(8)}
        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - float(np.dot(X[i], X[j]))
                if y[i] == y[j]:
                    same.append(d)
                    same_per_emo[int(y[i])].append(d)
                else:
                    diff.append(d)
                    diff_per_emo[int(y[i])].append(d)
                    diff_per_emo[int(y[j])].append(d)
        intra = float(np.mean(same)) if same else 0.0
        inter = float(np.mean(diff)) if diff else 0.0
        per_emo_intra = {i: (float(np.mean(v)) if v else 0.0) for i, v in same_per_emo.items()}
        return intra, inter, per_emo_intra

    intra_c, inter_c, per_emo_c = _intra_inter(Xc, labels)
    intra_n, inter_n, per_emo_n = _intra_inter(Xn, labels)

    print('\n[metric] cosine distance (smaller = closer)')
    print(f'  cond: intra={intra_c:.4f}  inter={inter_c:.4f}  ratio={intra_c/(inter_c+1e-9):.3f}')
    print(f'  norm: intra={intra_n:.4f}  inter={inter_n:.4f}  ratio={intra_n/(inter_n+1e-9):.3f}')
    print('  per-emo intra (cond):', {k: round(v, 4) for k, v in per_emo_c.items()})
    print('  per-emo intra (norm):', {k: round(v, 4) for k, v in per_emo_n.items()})

    # 指标也写到文件
    metric_path = out_dir / 'cluster_metrics.txt'
    with open(metric_path, 'w', encoding='utf-8') as f:
        f.write('Cosine-distance clustering metrics (1 - cos_sim, lower = closer)\n')
        f.write(f'cond : intra={intra_c:.6f}  inter={inter_c:.6f}  ratio={intra_c/(inter_c+1e-9):.4f}\n')
        f.write(f'norm : intra={intra_n:.6f}  inter={inter_n:.6f}  ratio={intra_n/(inter_n+1e-9):.4f}\n')
        f.write('\nper-emo intra (cond):\n')
        for k, v in per_emo_c.items():
            f.write(f'  {_EMO_LIST_ORIG[k]:>10s} : {v:.6f}\n')
        f.write('\nper-emo intra (norm):\n')
        for k, v in per_emo_n.items():
            f.write(f'  {_EMO_LIST_ORIG[k]:>10s} : {v:.6f}\n')
    print(f'[save] metrics -> {metric_path}')

    # 把降维结果也存下来
    np.savez(out_dir / 'embeddings.npz',
             labels=labels,
             Xc=Xc, Xn=Xn,
             umap_cond=emb_c_umap, umap_norm=emb_n_umap,
             tsne_cond=emb_c_tsne, tsne_norm=emb_n_tsne,
             emo_list=np.array(_EMO_LIST_ORIG))
    print(f'[save] embeddings -> {out_dir / "embeddings.npz"}')


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/emo_dit/20260815_fuke_20260806_emotion_dit_Unification_bs128_lexp1/checkpoints/iter_0300000.pt')
    parser.add_argument('--data_root', type=str, default='src/my_prepare')
    parser.add_argument('--motion_filename', type=str, default='front_all_motions.pkl')
    parser.add_argument('--motion_template_filename', type=str, default='motion_template.pkl')
    parser.add_argument('--n_per_emo', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--perplexity', type=int, default=20)
    parser.add_argument('--umap_neighbors', type=int, default=15)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip_features', action='store_true',
                        help='如果已经存过 features_cond.npy / features_norm.npy / labels.npy，跳过模型前向')
    args = parser.parse_args()

    os.chdir(ROOT)  # 让相对路径生效
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    out_dir = Path(args.out_dir) if args.out_dir \
        else (ROOT / 'experiments' / 'emo_dit' / '20260815_fuke_20260806_emotion_dit_Unification_bs128_lexp1' / 'visualizations')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[out_dir] {out_dir}')

    feat_cond_path = out_dir / 'features_cond.npy'
    feat_norm_path = out_dir / 'features_norm.npy'
    lbl_path = out_dir / 'labels.npy'

    if args.skip_features and feat_cond_path.exists() and feat_norm_path.exists() and lbl_path.exists():
        print('[skip_features] loading cached features ...')
        feats_cond = np.load(feat_cond_path)
        feats_norm = np.load(feat_norm_path)
        labels = np.load(lbl_path)
    else:
        # ---- 1) 加载模型 ----
        print('[1/3] building model from checkpoint ...')
        model, ckpt = build_model_from_ckpt(args.checkpoint, device)
        print(f'  checkpoint iter = {ckpt.get("iter")}')

        # ---- 2) 加载训练集 ----
        print('[2/3] loading training set ...')
        base_ds = EmoLevelDataset(
            root_dir=args.data_root,
            motion_filename=args.motion_filename,
            motion_template_filename=args.motion_template_filename,
            split='train',
            coef_fps=25,
            n_motions=model.n_motions,
            n_prev_motions=model.n_prev_motions,
            crop_strategy='begin',  # 复现用，避免 random 切片带来的随机性
            normalize_type='mix',
        )
        sub = BalancedSubset(base_ds, n_per_emo=args.n_per_emo, seed=args.seed)
        print(f'  balanced subset size: {len(sub)} '
              f'({args.n_per_emo} per emotion)')

        loader = data.DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        # ---- 3) 提取特征 ----
        print('[3/3] extracting audio_feat_cond ...')
        feats_cond, feats_norm, labels = extract_audio_feat_cond(
            model, loader, device, audio_unit=base_ds.audio_unit,
        )
        print(f'  feats_cond {feats_cond.shape}, feats_norm {feats_norm.shape}, labels {labels.shape}')

        np.save(feat_cond_path, feats_cond)
        np.save(feat_norm_path, feats_norm)
        np.save(lbl_path, labels)
        print(f'[save] features -> {out_dir}')

    # ---- 4) 降维 + 画图 ----
    print('[viz] reducing + plotting ...')
    reduce_and_plot(feats_cond, feats_norm, labels, out_dir,
                    perplexity=args.perplexity,
                    umap_neighbors=args.umap_neighbors,
                    seed=args.seed)
    print('[done]')


if __name__ == '__main__':
    main()
