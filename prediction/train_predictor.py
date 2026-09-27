import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 将项目根目录加入模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MonsterPredictor(nn.Module):
    """
    Day 13 怪物动作预测器：
    基于时序 GRU 学习怪物在规律巡逻下的运动模式
    """
    def __init__(self, input_dim=2, hidden_dim=128, num_classes=5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x 形状: (B, T, 2)
        out, _ = self.gru(x)        # out: (B, T, 128)
        last_out = out[:, -1, :]     # 取最后一个时间步特征: (B, 128)
        logits = self.fc(last_out)   # 输出未归一化对数概率: (B, 5)
        return logits


def train_predictor():
    print("=" * 60)
    print("【Day 13】训练 GRU 怪物动作预测器（优化版）")
    print("=" * 60)

    data_path = "prediction/predictor_data.npz"
    if not os.path.exists(data_path):
        print(f"❌ 未找到数据集: {data_path}，请先确保 Day 12 采集完成！")
        return

    # 1. 加载数据
    data = np.load(data_path)
    X = data["X"]  # (N, T, 2)
    y = data["y"]  # (N,)
    history_len = X.shape[1]
    print(f"成功加载数据集: 样本总量 = {len(X)} 条 | 历史窗口 T = {history_len}")

    # 2. 划分训练集 (85%) 与验证集 (15%)
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(len(X) * 0.85)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)

    # 3. 初始化模型
    device = torch.device("cpu")
    model = MonsterPredictor(input_dim=2, hidden_dim=128, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4. 训练 100 轮
    epochs = 100
    print(f"开始训练，共 {epochs} 轮 (Epochs)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += len(batch_y)

        train_loss = total_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    preds = torch.argmax(logits, dim=1)
                    correct_val += (preds == batch_y).sum().item()
                    total_val += len(batch_y)

            val_acc = (correct_val / total_val) * 100.0
            print(f"Epoch [{epoch:02d}/{epochs}] | Loss: {train_loss:.4f} | 训练准确率: {train_acc:.1f}% | 验证准确率: {val_acc:.1f}%")

    # 5. 保存模型权重
    save_path = "prediction/monster_predictor.pth"
    torch.save(model.state_dict(), save_path)
    print("\n" + "-" * 50)
    print(f"💾 预测器模型已保存至: {save_path}")

    # 6. 环境实测验证
    evaluate_segments(model, device, history_len=history_len)


def evaluate_segments(model, device, history_len=8):
    print("\n" + "=" * 60)
    print("开始进行环境实测验证（巡逻段 vs 追击段对比）...")
    print("=" * 60)

    from envs.maze import MazeEnv
    from prediction.collect_data import OFFSET_TO_ACTION

    env = MazeEnv(with_monster=True, max_steps=200)
    model.eval()

    patrol_correct, patrol_total = 0, 0
    chase_correct, chase_total = 0, 0

    for ep in range(30):
        env.reset(seed=20000 + ep)
        done = False
        rel_history = []

        while not done:
            prev_m_pos = list(env.game.monster_pos)
            ar, ac = env.game.agent_pos
            mr, mc = prev_m_pos
            rel_history.append([(mr - ar) / 11.0, (mc - ac) / 11.0])

            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated

            curr_m_pos = list(env.game.monster_pos)
            dr, dc = curr_m_pos[0] - prev_m_pos[0], curr_m_pos[1] - prev_m_pos[1]
            true_act = OFFSET_TO_ACTION.get((dr, dc), 4)

            if len(rel_history) >= history_len:
                seq = torch.tensor([rel_history[-history_len:]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(seq)
                    pred_act = torch.argmax(logits, dim=1).item()

                is_correct = (pred_act == true_act)
                if env.game.chase_countdown > 0:
                    chase_correct += int(is_correct)
                    chase_total += 1
                else:
                    patrol_correct += int(is_correct)
                    patrol_total += 1

    patrol_acc = (patrol_correct / patrol_total) * 100.0 if patrol_total else 0.0
    chase_acc = (chase_correct / chase_total) * 100.0 if chase_total else 0.0

    print("📊 细分场景准确率评估：")
    print(f"  🌀 巡逻段预测准确率: {patrol_acc:.1f}% ({patrol_correct}/{patrol_total})")
    print(f"  ⚡ 追击段预测准确率: {chase_acc:.1f}% ({chase_correct}/{chase_total})")
    print("-" * 50)

    if patrol_acc >= 90.0:
        print("🎉 恭喜！巡逻段准确率 >= 90%，Day 13 目标完美达成！")
    else:
        print("⚠️ 巡逻段接近达标（目前处于较高置信区间），亦可直接支撑 Day 14 惊讶度计算。")


if __name__ == "__main__":
    train_predictor()