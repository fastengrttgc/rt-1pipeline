'''
2026/9/3 wangyuqi

'''

"""
smoke_train.py —— 阶段3 验收入口
把 数据管线(dataset) + 动作离散化(action_tokenizer) + 损失(loss) + 模型 串起来，
跑几个 batch 验证：shape 正确、前向通、反向通、优化器能 step。
"""
import torch

from dataset import load_episodes, split_episodes, RT1Dataset, collate_fn
from action_tokenizer import ActionTokenizer
from loss import RT1Loss, top1_action_accuracy

from robotic_transformer_pytorch import MaxViT, RT1

# ---------- 配置 ----------
DATA_DIR = None          # None 则用 dataset.py 里的默认路径
BATCH_SIZE = 2
NUM_SMOKE_STEPS = 3      # 冒烟只跑 3 步
LR = 1e-4
SEED = 0

torch.manual_seed(SEED)


def main():
    # 1. 预加载 + 切分（跑一次，约十几秒）
    episodes = load_episodes(DATA_DIR)
    train_eps, val_eps = split_episodes(episodes)
    print(f"train {len(train_eps)} 条 / val {len(val_eps)} 条")

    # 2. tokenizer 只在【训练集】上 fit（val 不参与统计）
    all_actions = torch.cat([torch.from_numpy(e["actions"]) for e in train_eps])  # (N, 7)
    tokenizer = ActionTokenizer(action_dim=7, num_bins=256).fit(all_actions)
    print("动作范围 min:", tokenizer.action_min.numpy().round(3))
    print("动作范围 max:", tokenizer.action_max.numpy().round(3))

    # 3. Dataset + DataLoader
    train_ds = RT1Dataset(train_eps)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    print(f"训练窗口总数: {len(train_ds)}")

    # 4. 模型（num_actions=7，与 Bridge 的 7 维动作对齐）
    vit = MaxViT(
        num_classes=1000, dim_conv_stem=64, dim=96, dim_head=32,
        depth=(2, 2, 5, 2), window_size=7,
        mbconv_expansion_rate=4, mbconv_shrinkage_rate=0.25, dropout=0.1,
    )
    model = RT1(
        vit=vit, num_actions=7, depth=6, heads=8, dim_head=64, cond_drop_prob=0.2,
    )
    model.train()

    criterion = RT1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 5. 冒烟训练几步
    for step, batch in enumerate(loader):
        if step >= NUM_SMOKE_STEPS:
            break

        video = batch["video"]      # (B, 3, 6, 224, 224)
        texts = batch["texts"]      # list[str]
        actions = batch["actions"]  # (B, 6, 7) 连续动作

        labels = tokenizer.encode(actions)   # (B, 6, 7) long，离散标签
        logits = model(video, texts)         # (B, 6, 7, 256)
        loss = criterion(logits, labels)
        acc = top1_action_accuracy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"step {step}: loss={loss.item():.4f}  top1_acc={acc.item():.6f}  "
              f"logits={tuple(logits.shape)}")

    print("冒烟训练完成 ✅ 阶段3 验收通过（无报错即成功）")


if __name__ == "__main__":
    main()