import tensorflow_datasets as tfds

builder = tfds.builder_from_directory(builder_dir="data/bridge_orig_ep100/1.0.0")
print("划分:", builder.info.splits)

ds = builder.as_dataset(split="train")

for ep in ds.take(1):                     # 取 1 条轨迹
    print("episode 字段:", list(ep.keys()))
    steps = ep["steps"]
    num = sum(1 for _ in steps)
    print("该轨迹帧数:", num)
    for step in steps.take(1):            # 看第一帧
        for k, v in step.items():
            if hasattr(v, "shape"):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {v}")