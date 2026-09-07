'''
2026/9/3 wangyuqi

'''

"""
loss.py —— RT-1 训练损失
职责：
  RT1Loss:          6 帧 × 7 动作维的 CrossEntropy（标签 = ActionTokenizer.encode 的结果）
  top1_action_accuracy: 预测 bin == 真实 bin 的比例（训练时顺手打印用）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RT1Loss(nn.Module):
    """对模型输出的每一帧、每一动作维分别做 256 类分类，取平均。

    logits: (B, F, D, num_bins) float   ← 模型输出（F=6 帧, D=7 动作维）
    labels: (B, F, D)          long     ← ActionTokenizer.encode 后的 bin 索引
    返回: 标量 loss
    """

    def forward(self, logits, action_labels):
        if logits.ndim != 4:
            raise ValueError(f"logits 需要 (B,F,D,num_bins)，收到 {tuple(logits.shape)}")
        if action_labels.ndim != 3:
            raise ValueError(f"action_labels 需要 (B,F,D)，收到 {tuple(action_labels.shape)}")

        if logits.shape[:3] != action_labels.shape:
            raise ValueError(
                f"shape 不匹配: logits {tuple(logits.shape)} vs labels {tuple(action_labels.shape)}"
            )

        # 注意：变量名不要用 F，它会覆盖 torch.nn.functional 的别名！
        B, n_frames, D, num_bins = logits.shape

        # 压平成标准的 (N, num_bins) + (N,) 分类问题
        logits_flat = logits.reshape(-1, num_bins)   # (B*F*D, 256)
        labels_flat = action_labels.reshape(-1)      # (B*F*D,)

        return F.cross_entropy(logits_flat, labels_flat)  # 默认对 N 求平均


def top1_action_accuracy(logits, action_labels):
    """预测的 bin 是否等于真实 bin（越接近 1 越好）。
    随机猜测时约为 1/256 ≈ 0.4%，可作为 sanity check。
    """
    preds = logits.argmax(dim=-1)          # (B, F, D)
    correct = (preds == action_labels).float()
    return correct.mean()


# ---------- 自测入口 ----------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, n_frames, D, num_bins = 2, 6, 7, 256   # 也不要用 F 做变量名

    logits = torch.randn(B, n_frames, D, num_bins, requires_grad=True)  # 随机 logits
    labels = torch.randint(0, num_bins, (B, n_frames, D))               # 随机标签

    criterion = RT1Loss()
    loss = criterion(logits, labels)
    acc = top1_action_accuracy(logits, labels)

    # 反向传播必须能通
    loss.backward()
    grad_ok = logits.grad is not None and torch.isfinite(logits.grad).all().item()

    # 和手写 F.cross_entropy 对比，验证实现正确
    manual = F.cross_entropy(logits.reshape(-1, num_bins), labels.reshape(-1))
    match = torch.allclose(loss, manual)

    print("loss        :", round(loss.item(), 4))
    print("top-1 准确率 :", round(acc.item(), 4), "(随机应为 ~0.004)")
    print("backward 通过 :", grad_ok)
    print("与手写一致    :", match)

    # 形状不匹配时要报错
    try:
        criterion(torch.randn(B, n_frames, D, num_bins), torch.randint(0, num_bins, (B, n_frames)))
        print("形状检查      : 失败（没报错）")
    except ValueError:
        print("形状检查      : 通过（正确报错）")