"""
train.py —— 阶段4 正式训练脚本
基于 rt1_pipeline 的 dataset / action_tokenizer / loss，训练 RT-1 (num_actions=7)。

用法示例:
  # 本地快速验证（CPU，几十步）
  HF_HUB_OFFLINE=1 python -u train.py --steps 20 --batch-size 2 --eval-freq 10 --save-freq 20

  # 云上正式训练（CUDA）
  HF_HUB_OFFLINE=1 python -u train.py --steps 2000 --batch-size 8 \
      --eval-freq 100 --save-freq 500 --checkpoint-dir checkpoints

  # 断点续训（--steps 是"绝对步数上限"：resume 到 1000 后再传 3000 会继续训到 3000）
  HF_HUB_OFFLINE=1 python -u train.py --resume checkpoints/checkpoint_step1000.pt --steps 3000

注意:
  - device=auto 时选 cuda，其次 cpu。MPS 目前与模型内置 T5 有设备兼容问题，未默认启用。
  - 66 条轨迹很小，训练多轮会过拟合，属正常；本阶段目标是链路完整 + loss 下降。
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from dataset import load_episodes, split_episodes, RT1Dataset, collate_fn
from action_tokenizer import ActionTokenizer
from loss import RT1Loss, top1_action_accuracy
from robotic_transformer_pytorch import MaxViT, RT1


# ----------------------------------------------------------------------
# 模型结构（训练与恢复必须完全一致）
# ----------------------------------------------------------------------

def build_model():
    vit = MaxViT(
        num_classes=1000, dim_conv_stem=64, dim=96, dim_head=32,
        depth=(2, 2, 5, 2), window_size=7,
        mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, dropout=0.1,
    )
    model = RT1(
        vit=vit, num_actions=7, depth=6, heads=8, dim_head=64, cond_drop_prob=0.2,
    )
    return model


# ----------------------------------------------------------------------
# 参数
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RT-1 正式训练")
    p.add_argument("--steps", type=int, default=500, help="绝对步数上限")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--eval-freq", type=int, default=50, help="每 N 步在验证集评估一次")
    p.add_argument("--save-freq", type=int, default=200, help="每 N 步存一次 checkpoint")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="从某个 .pt checkpoint 续训")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------
# 验证集评估
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, tokenizer, criterion, device, max_batches=10):
    """返回 (平均 eval_loss, 平均 top1_acc)"""
    model.eval()
    losses, accs = [], []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        video = batch["video"].to(device)
        texts = batch["texts"]
        labels = tokenizer.encode(batch["actions"]).to(device)
        logits = model(video, texts)
        losses.append(criterion(logits, labels).item())
        accs.append(top1_action_accuracy(logits, labels).item())
    model.train()
    return float(np.mean(losses)), float(np.mean(accs))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 设备选择：auto = cuda 优先，其次 cpu（mps 需先解决 T5 兼容问题）
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---------- 1. 数据 ----------
    episodes = load_episodes()
    train_eps, val_eps = split_episodes(episodes, val_ratio=args.val_ratio, seed=args.seed)
    print(f"train {len(train_eps)} 条 / val {len(val_eps)} 条")

    # ---------- 2. 动作离散化：只在训练集上 fit，边界落盘（部署必须用同一组） ----------
    all_actions = torch.cat([torch.from_numpy(e["actions"]) for e in train_eps])
    tokenizer = ActionTokenizer(action_dim=7, num_bins=256).fit(all_actions)
    bounds_path = os.path.join(args.checkpoint_dir, "action_bounds.npz")
    tokenizer.save(bounds_path)
    print(f"动作边界已保存: {bounds_path}")

    # ---------- 3. DataLoader ----------
    train_ds = RT1Dataset(train_eps)
    val_ds = RT1Dataset(val_eps)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    print(f"训练窗口总数: {len(train_ds)}，验证窗口总数: {len(val_ds)}")

    # ---------- 4. 模型 / 优化器 / 损失 ----------
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = RT1Loss()

    # ---------- 5. 恢复（resume） ----------
    start_step = 0
    best_eval = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_eval = ckpt.get("best_eval_loss", float("inf"))
        print(f"已从 {args.resume} 恢复，从 step {start_step} 继续到 {args.steps}")

    model.train()

    # ---------- 6. CSV 日志 ----------
    log_path = os.path.join(args.checkpoint_dir, "train_log.csv")
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        writer.writerow(["step", "train_loss", "eval_loss", "eval_acc"])
        log_file.flush()

    # ---------- 7. 训练循环 ----------
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.1f}M")
    print(f"开始训练，共 {args.steps} 步...")

    step = start_step
    t_start = time.time()

    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break

            video = batch["video"].to(device)
            texts = batch["texts"]
            labels = tokenizer.encode(batch["actions"]).to(device)

            logits = model(video, texts)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # 进度打印
            if step % 10 == 0 or step >= args.steps:
                elapsed = time.time() - t_start
                sec_per_step = elapsed / max(1, step - start_step)
                print(f"step {step}/{args.steps}  loss={loss.item():.4f}  "
                      f"({sec_per_step:.2f}s/step)", flush=True)

            # 周期评估
            if args.eval_freq and step % args.eval_freq == 0:
                eval_loss, eval_acc = evaluate(model, val_loader, tokenizer,
                                               criterion, device)
                writer.writerow([step, round(loss.item(), 4),
                                 round(eval_loss, 4), round(eval_acc, 4)])
                log_file.flush()
                print(f"  [eval] step {step}: eval_loss={eval_loss:.4f} "
                      f"eval_acc={eval_acc:.4f}")

                # 跟踪最优模型
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    best_path = os.path.join(args.checkpoint_dir, "best.pt")
                    torch.save(model.state_dict(), best_path)   # 只存权重(~1.1GB)
                    print(f"  [best] 新最优 eval_loss={eval_loss:.4f}，已存 {best_path}")

            # 周期保存（可续训的 checkpoint）
            if args.save_freq and step % args.save_freq == 0:
                ckpt_path = os.path.join(
                    args.checkpoint_dir, f"checkpoint_step{step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_eval_loss": best_eval,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  [save] 已存 {ckpt_path}")

    # ---------- 8. 收尾 ----------
    final_path = os.path.join(args.checkpoint_dir, f"checkpoint_step{step}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_eval_loss": best_eval,
        "args": vars(args),
    }, final_path)
    torch.save(model.state_dict(),
               os.path.join(args.checkpoint_dir, "final_model.pt"))
    print("已存 final_model.pt（仅权重，供阶段5/6 评估与部署）")
    log_file.close()
    total_min = (time.time() - t_start) / 60
    print(f"训练完成 ✅ 共 {step - start_step} 步，用时 {total_min:.1f} 分钟")
    print(f"最终 checkpoint: {final_path}")
    print(f"训练日志: {log_path}")


if __name__ == "__main__":
    main()