import json

import matplotlib.pyplot as plt

# 设置风格
plt.style.use('ggplot')


def plot_training_results(log_path='checkpoints/training_log.json'):
    try:
        with open(log_path, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print("未找到日志文件，请先运行训练脚本。")
        return

    epochs = history['epoch']
    loss = history['loss']
    train_acc = history['train_acc']
    test_acc = history['test_acc']

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 图1：损失曲线
    ax1.plot(epochs, loss, label='Training Loss', color='tab:red', linewidth=2)
    ax1.set_title('Training Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 图2：准确率曲线
    ax2.plot(epochs, train_acc, label='Train Accuracy', color='tab:blue', linestyle='--')
    ax2.plot(epochs, test_acc, label='Test Accuracy', color='tab:green', linewidth=2)
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # 标注最高测试准确率
    max_acc = max(test_acc)
    max_epoch = epochs[test_acc.index(max_acc)]
    ax2.annotate(f'Best: {max_acc:.2f}%', xy=(max_epoch, max_acc), xytext=(max_epoch, max_acc - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig('checkpoints/training_results.png', dpi=300)
    print("图表已保存为 checkpoints/training_results.png")
    plt.show()


if __name__ == '__main__':
    plot_training_results()