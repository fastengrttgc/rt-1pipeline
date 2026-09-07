'''
2026/9/3 wangyuqi

'''

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import tensorflow_datasets as tfds

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bridge_orig_ep100" / "1.0.0"

# RT-1 的窗口约定
WINDOW_FRAMES = 6     # 每窗口 6 帧
FRAME_STRIDE = 3      # 每 3 帧取 1 帧
WINDOW_STRIDE = 3     # 窗口滑动步长（越大样本越少，可调）

def load_episodes(builder_dir=None):
    """把 TFDS 数据一次性读进内存。
    返回 list[dict]，每个 dict:
      images:      uint8   (T, 224, 224, 3)  全部帧
      actions:     float32 (T, 7)             全部动作（连续值，离散化留给 tokenizer）
      instruction: str                        该轨迹的指令
    """
    builder_dir = builder_dir or str(DATA_DIR)
    builder = tfds.builder_from_directory(builder_dir=builder_dir)
    ds = builder.as_dataset(split="train")

    episodes = []
    for ep in ds:
        steps = list(ep["steps"])
        episodes.append({
            "images": np.stack([s["observation"]["image"].numpy() for s in steps]),
            "actions": np.stack([s["action"].numpy() for s in steps]),
            "instruction": steps[0]["language_instruction"].numpy().decode("utf-8"),
        })

    # 输出获取的数据
    lens = [len(e["images"]) for e in episodes]
    n_empty = sum(1 for e in episodes if not e["instruction"].strip())
    print(f"[load_episodes] 共 {len(episodes)} 条轨迹, 帧数 {min(lens)}-{max(lens)}, "
          f"空指令轨迹 {n_empty} 条")
    return episodes

# train/val切分
def split_episodes(episodes, val_ratio=0.1, seed=0):
    """按轨迹比例切分，返回 (train_episodes, val_episodes)"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [episodes[i] for i in train_idx], [episodes[i] for i in val_idx]

# 滑动窗口采样
class RT1Dataset(Dataset):
    def __init__(self, episodes, window_frames=WINDOW_FRAMES,
                 frame_stride=FRAME_STRIDE, window_stride=WINDOW_STRIDE,
                 drop_empty_instruction=True):
        self.episodes = episodes
        self.window_frames = window_frames
        self.frame_stride = frame_stride

        # 预计算所有合法窗口：记录 (轨迹下标, 起始帧)
        self.samples = []
        for ep_idx, ep in enumerate(episodes):
            if drop_empty_instruction and not ep["instruction"].strip():
                continue
            T = len(ep["images"])
            max_start = T - 1 - (window_frames - 1) * frame_stride  # 窗口最后一帧 ≤ T-1
            if max_start < 0:
                continue  # 轨迹太短，装不下一个窗口，丢弃
            for start in range(0, max_start + 1, window_stride):
                self.samples.append((ep_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, start = self.samples[idx]
        ep = self.episodes[ep_idx]

        # 6 个采样帧下标：start, start+3, ..., start+15（时间有序，旧→新）
        f_idx = start + np.arange(self.window_frames) * self.frame_stride

        images = ep["images"][f_idx]    # (6, 224, 224, 3) uint8
        actions = ep["actions"][f_idx]  # (6, 7) float32，每帧的动作 = 该帧的监督标签

        # uint8 → float [0,1]，重排成 (C, F, H, W) = (3, 6, 224, 224)
        video = torch.from_numpy(images).permute(3, 0, 1, 2).float() / 255.0

        return {
            "video": video,                          # (3, 6, 224, 224)
            "texts": ep["instruction"],              # str（批量后变成 list[str]）
            "actions": torch.from_numpy(actions),    # (6, 7) 连续动作，离散化在训练入口做
        }

def collate_fn(batch):
    """把一批样本堆叠成 batch；指令是字符串，默认 collate 拼不了，必须自定义"""
    videos = torch.stack([b["video"] for b in batch])          # (B, 3, 6, 224, 224)
    actions = torch.stack([b["actions"] for b in batch])       # (B, 6, 7)
    texts = [b["texts"] for b in batch]                        # list[str]
    return {"video": videos, "texts": texts, "actions": actions}

# ---------- 自测入口（单独运行本文件时执行） ----------

if __name__ == "__main__":
    episodes = load_episodes()
    train_eps, val_eps = split_episodes(episodes)
    print(f"[split] train {len(train_eps)} 条 / val {len(val_eps)} 条")

    ds = RT1Dataset(train_eps)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("video  :", tuple(batch["video"].shape))     # 预期 (4, 3, 6, 224, 224)
    print("actions:", tuple(batch["actions"].shape))   # 预期 (4, 6, 7)
    print("texts  :", batch["texts"])