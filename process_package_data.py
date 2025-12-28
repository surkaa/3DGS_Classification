import glob
import os

import numpy as np
import torch
import tqdm
from plyfile import PlyData

ROOT_DIR = r"W:\Datasets\MACGS"
SAVE_PATH = "macgs_packed_v2.pt"
NUM_POINTS = 2048
NUM_CLASSES = 30


def load_and_process_ply(path):
    try:
        plydata = PlyData.read(path)
        v = plydata['vertex']

        # 1. 基础坐标
        x = np.asarray(v['x'])
        y = np.asarray(v['y'])
        z = np.asarray(v['z'])

        # 2. 颜色处理 (兼容性读取)
        try:
            r = np.asarray(v['f_dc_0'])
            g = np.asarray(v['f_dc_1'])
            b = np.asarray(v['f_dc_2'])
        except:
            r = np.asarray(v['red']) / 255.0
            g = np.asarray(v['green']) / 255.0
            b = np.asarray(v['blue']) / 255.0

        # 3. === 读取 3DGS 特有属性 ===
        # 3DGS 中 opacity 通常存为 logit，需要 sigmoid 归一化到 0-1
        try:
            op_raw = np.asarray(v['opacity'])
            opacity = 1 / (1 + np.exp(-op_raw))
        except:
            # 如果是普通点云没有opacity，默认全为1 (实心)
            opacity = np.ones_like(x)

        # 3DGS 中 scale 通常存为 log scale，需要 exp 还原
        try:
            s0 = np.exp(np.asarray(v['scale_0']))
            s1 = np.exp(np.asarray(v['scale_1']))
            s2 = np.exp(np.asarray(v['scale_2']))
        except:
            # 如果没有 scale，默认为很小的球
            s0 = np.zeros_like(x) + 0.01
            s1 = np.zeros_like(x) + 0.01
            s2 = np.zeros_like(x) + 0.01

        # 组合特征: [N, 10] -> x,y,z, r,g,b, op, sx,sy,sz
        points = np.stack([x, y, z, r, g, b, opacity, s0, s1, s2], axis=1)

        # === 采样 (2048 点) ===
        # 即使不过滤透明点，增加采样数也能保证采到核心物体
        if len(points) >= NUM_POINTS:
            choice = np.random.choice(len(points), NUM_POINTS, replace=False)
        else:
            choice = np.random.choice(len(points), NUM_POINTS, replace=True)
        points = points[choice, :]

        # === 归一化 (仅对 XYZ 做几何归一化) ===
        xyz = points[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        # 缩放到单位球内
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1))) + 1e-8
        xyz = xyz / m
        points[:, :3] = xyz

        points[:, 7:10] = points[:, 7:10] / m

        # 转为 Tensor: (10, 2048)
        return torch.from_numpy(points).float().transpose(0, 1)

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def pack_dataset():
    packed_data = {'train': [], 'test': []}
    all_files = []

    # 扫描文件
    print("Scanning files...")
    for label in range(NUM_CLASSES):
        folder_path = os.path.join(ROOT_DIR, str(label))
        if not os.path.exists(folder_path): continue
        ply_files = glob.glob(os.path.join(folder_path, "*.ply"))
        for p in ply_files:
            all_files.append((p, label))

    # 打乱
    import random
    random.seed(42)
    random.shuffle(all_files)

    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]

    print(f"Total: {len(all_files)} | Train: {len(train_files)} | Test: {len(test_files)}")

    for path, label in tqdm.tqdm(train_files, desc="Packing Train"):
        tensor = load_and_process_ply(path)
        if tensor is not None: packed_data['train'].append((tensor, label))

    for path, label in tqdm.tqdm(test_files, desc="Packing Test"):
        tensor = load_and_process_ply(path)
        if tensor is not None: packed_data['test'].append((tensor, label))

    print("Saving...")
    torch.save(packed_data, SAVE_PATH)
    print(f"Done. Size: {os.path.getsize(SAVE_PATH) / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    pack_dataset()
