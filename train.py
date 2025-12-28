import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import Dataset, DataLoader
from model import DGCNN

DATA_PATH = "macgs_packed_v2.pt"
BATCH_SIZE = 24
EPOCHS = 200
LEARNING_RATE = 0.001
NUM_CLASSES = 30


# === 工具类：早停机制 ===
class EarlyStopping:
    def __init__(self, patience=15, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_acc = 0

    def __call__(self, val_acc, model, path):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, path)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, path):
        print(f'Validation accuracy improved ({self.best_acc:.2f}% --> {val_acc:.2f}%).  Saving model...')
        torch.save(model.state_dict(), path)
        self.best_acc = val_acc


class CachedDataset(Dataset):
    def __init__(self, data_list, split='train'):
        self.data_list = data_list
        self.split = split

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # data_tensor shape: (10, 2048) -> x,y,z,r,g,b,op,sx,sy,sz
        data_tensor, label = self.data_list[idx]

        if self.split == 'train':
            # 转为 numpy: (10, N) -> (N, 10)
            data = data_tensor.numpy().T.copy()

            xyz = data[:, :3]
            features = data[:, 3:]  # (N, 7) 包括颜色、透明度、缩放

            # 1. 随机旋转 (只旋转 XYZ)
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            xyz = np.dot(xyz, rotation_matrix)

            # 2. 随机抖动 (只抖动 XYZ)
            xyz += np.random.normal(0, 0.005, size=xyz.shape)
            # 缩放也可以轻微抖动，模拟不同尺度的噪声
            # features[:, 4:] *= np.random.uniform(0.9, 1.1)

            data = np.concatenate([xyz, features], axis=1)
            return torch.from_numpy(data).float().transpose(0, 1), label

        return data_tensor, label


if __name__ == '__main__':
    # 检查文件夹
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading packed dataset from {DATA_PATH}...")
    try:
        loaded_data = torch.load(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        exit()

    train_data = loaded_data['train']
    test_data = loaded_data['test']
    print(f"Loaded! Train: {len(train_data)}, Test: {len(test_data)}")

    # Dataset & DataLoader
    train_dataset = CachedDataset(train_data, split='train')
    test_dataset = CachedDataset(test_data, split='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model, Optimizer, Loss
    model = DGCNN(num_classes=NUM_CLASSES, k=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Early Stopping & Logging
    early_stopping = EarlyStopping(patience=20, delta=0.01)
    save_path = 'checkpoints/best_model_v2.pth'

    # === 新增：用于记录数据的字典 ===
    history = {
        'epoch': [],
        'loss': [],
        'train_acc': [],
        'test_acc': []
    }

    print(f"\nStart Training (Max {EPOCHS} Epochs)...")

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        total_loss = 0

        pbar = tqdm.tqdm(train_loader, desc=f"Ep {epoch + 1}/{EPOCHS}", unit="bt")
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'Loss': f"{loss.item():.2f}"})

        scheduler.step()

        # 计算本轮指标
        epoch_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Test
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test

        print(f"Result: Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # === 记录数据 ===
        history['epoch'].append(epoch + 1)
        history['loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        # 实时保存到 JSON，防止中断丢失
        with open('checkpoints/training_log.json', 'w') as f:
            json.dump(history, f, indent=4)

        # 早停检查
        early_stopping(test_acc, model, save_path)
        if early_stopping.early_stop:
            print("Early stopping triggered! Training finished.")
            break

    print(f"Final Best Test Accuracy: {early_stopping.best_acc:.2f}%")
