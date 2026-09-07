'''
2026/9/3 wangyuqi

'''

"""
action_tokenizer.py —— 动作离散化（RT-1 风格：每维 256 bins）
职责：
  fit():    用【训练集】统计每维动作的 min/max（归一化边界，测试/部署必须一致）
  encode(): 连续动作 → bin 索引（训练标签）
  decode(): bin 索引 → 连续动作（bin 中心，部署时用）
"""
import numpy as np
import torch


class ActionTokenizer:
    def __init__(self, action_dim=7, num_bins=256, eps=1e-6):
        self.action_dim = action_dim
        self.num_bins = num_bins      # 和模型的 action_bins=256 一致
        self.eps = eps
        self.action_min = None        # (D,) 每维最小值
        self.action_max = None        # (D,) 每维最大值

    # ---------- 归一化边界 ----------

    def fit(self, actions):
        """actions: (N, D) 连续动作（只传【训练集】的动作，别让 val 泄漏统计量）"""
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        if actions.ndim != 2:
            raise ValueError(f"fit 需要 (N, D) 形状，收到 {tuple(actions.shape)}")
        self.action_min = actions.min(dim=0).values   # (D,)
        self.action_max = actions.max(dim=0).values
        return self

    # ---------- 连续 → bin ----------

    def encode(self, actions):
        """连续动作 → bin 索引。
        输入任意前导维 (..., D)，返回同形状的 long，范围 [0, num_bins-1]
        """
        self._check_fitted()
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()

        norm = (actions - self.action_min) / (self.action_max - self.action_min + self.eps)
        norm = norm.clamp(0.0, 1.0)                          # 防越界
        bins = (norm * self.num_bins).long().clamp(0, self.num_bins - 1)
        return bins                                          # (..., D) long

    # ---------- bin → 连续 ----------

    def decode(self, labels):
        """bin 索引 → 连续动作。
        用 bin 中心 (label + 0.5)/num_bins 还原，比用左边界量化误差小一半。
        """
        self._check_fitted()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        norm = (labels.float() + 0.5) / self.num_bins        # 取每个 bin 的中心
        actions = self.action_min + norm * (self.action_max - self.action_min)
        return actions                                       # (..., D) float

    # ---------- 保存/加载边界（阶段6 部署必须用同一组边界） ----------

    def save(self, path):
        np.savez(path, action_min=self.action_min.numpy(),
                 action_max=self.action_max.numpy())

    def load(self, path):
        data = np.load(path)
        self.action_min = torch.from_numpy(data["action_min"]).float()
        self.action_max = torch.from_numpy(data["action_max"]).float()
        return self

    # ---------- 内部 ----------

    def _check_fitted(self):
        if self.action_min is None:
            raise RuntimeError("先调用 fit() 再使用 encode/decode")


# ---------- 自测入口 ----------

if __name__ == "__main__":
    torch.manual_seed(0)
    # 模拟一组动作：每维范围不同（比如 [-0.5,0.5], [-1,1] 混合）
    D = 7
    lows  = torch.tensor([-0.5, -1.0, -0.3, -0.8, -1.0, -0.5,  0.0])
    highs = torch.tensor([ 0.5,  1.0,  0.3,  0.8,  1.0,  0.5,  1.0])
    train_actions = (highs - lows) * torch.rand(2000, D) + lows

    tok = ActionTokenizer(action_dim=D).fit(train_actions)
    print("action_min:", tok.action_min.numpy().round(3))
    print("action_max:", tok.action_max.numpy().round(3))

    # 往返测试
    labels = tok.encode(train_actions)
    recovered = tok.decode(labels)
    err = (recovered - train_actions).abs()
    print("标签范围:", labels.min().item(), "-", labels.max().item())
    print("每维最大往返误差:", err.max(dim=0).values.numpy().round(4))
    print("理论最大误差(半个 bin):", ((highs - lows) / 256 / 2).numpy().round(4))