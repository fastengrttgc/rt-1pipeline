"""benchmark.py —— 测训练速度（平均每步耗时）"""
import time, sys
import torch
from dataset import load_episodes, split_episodes, RT1Dataset, collate_fn
from action_tokenizer import ActionTokenizer
from loss import RT1Loss
from robotic_transformer_pytorch import MaxViT, RT1

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
print("device:", device)
torch.manual_seed(0)

# ---- 数据（和训练一致）----
episodes = load_episodes()
train_eps, _ = split_episodes(episodes)
all_actions = torch.cat([torch.from_numpy(e["actions"]) for e in train_eps])
tokenizer = ActionTokenizer(action_dim=7).fit(all_actions)
ds = RT1Dataset(train_eps)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)

# ---- 模型 ----
vit = MaxViT(num_classes=1000, dim_conv_stem=64, dim=96, dim_head=32,
             depth=(2, 2, 5, 2), window_size=7, mbconv_expansion_rate=4,
             mbconv_shrinkage_rate=0.25, dropout=0.1)
model = RT1(vit=vit, num_actions=7, depth=6, heads=8, dim_head=64, cond_drop_prob=0.2)
model.to(device)
model.train()
criterion = RT1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def step(batch):
    labels = tokenizer.encode(batch["actions"]).to(device)
    logits = model(batch["video"].to(device), batch["texts"])
    loss = criterion(logits, labels)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# ---- 预热 1 步（MPS 首次分配内存慢，不计时）----
warmup = next(iter(loader))
step(warmup)

# ---- 计时 5 步 ----
N = 5
t0 = time.time()
for i, batch in enumerate(loader):
    if i >= N: break
    step(batch)
dt = (time.time() - t0) / N
print(f"平均每步: {dt:.2f} 秒")
print(f"估算: 500步≈{dt*500/60:.1f}分钟, 1000步≈{dt*1000/60:.1f}分钟, 2000步≈{dt*2000/60:.1f}分钟")