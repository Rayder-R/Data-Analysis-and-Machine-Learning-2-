
'''
# **一個基本的多層感知器（Multilayer Perceptron, MLP）用於二分類任務**。
'''

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# %% 資料準備
# 生成環形數據
np.random.seed(40)
n_points = 100

# 類 0: 圓形內部
theta = np.linspace(0, 2 * np.pi, n_points)
r_0 = 1 + 0.2 * np.random.randn(n_points)  # 加一些隨機擾動
class_0 = np.vstack([r_0 * np.cos(theta), r_0 * np.sin(theta)]).T

# 類 1: 圓形外部
r_1 = 2 + 0.2 * np.random.randn(n_points)  # 外圓
class_1 = np.vstack([r_1 * np.cos(theta), r_1 * np.sin(theta)]).T

print("class_0 shape:", class_0[:5], class_0.shape)
print("class_1 shape:", class_0[:5], class_1.shape)
# 合併資料與標籤
data = np.vstack([class_0, class_1]).astype(np.float32)
labels = np.array([0]*100 + [1]*100).astype(np.float32).reshape(-1, 1)

# 顯示合併後資料與 shape
print("Data set:", data[:1], "shape:", data.shape)
print("Labels set:", labels[:1], "shape:", labels.shape)

# 繪製數據
plt.scatter(class_0[:, 0], class_0[:, 1], color='#003060',
            label='Class 0', alpha=0.7, s=40)
plt.scatter(class_1[:, 0], class_1[:, 1], color='black',
            label='Class 1', alpha=0.3, s=40)
plt.legend()
plt.title("Circular Data")
plt.show()


# %% 定義神經網路
# 多層感知器 (Multilayer Perceptron, MLP) 設計
class MultilayerPerceptron:
    """
    一個基本的多層感知器模型，包含一個隱藏層和一個輸出層，用於二分類任務。
    """

    def __init__(self, input_dim, hidden_dim):
        # 隱藏層的權重和偏置
        self.W1 = tf.Variable(tf.random.normal([input_dim, hidden_dim]))
        self.b1 = tf.Variable(tf.zeros([hidden_dim]))
        # 輸出層的權重和偏置
        self.W2 = tf.Variable(tf.random.normal([hidden_dim, 1]))
        self.b2 = tf.Variable(tf.zeros([1]))

    def forward(self, x):
        # 隱藏層 + ReLU 激活函數
        hidden_output = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        return tf.matmul(hidden_output, self.W2) + self.b2

# 定義交叉熵損失函數


def binary_crossentropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))


# 訓練模型
def train_model(model, data, labels, learning_rate=0.1, epochs=100):
    optimizer = tf.optimizers.SGD(learning_rate)

    loss_history = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            logits = model.forward(data)  # 前向傳播
            loss = binary_crossentropy_loss(labels, logits)  # 計算損失

        # 計算梯度並更新權重
        gradients = tape.gradient(
            loss, [model.W1, model.b1, model.W2, model.b2])  # 更新所有權重和偏置
        optimizer.apply_gradients(
            zip(gradients, [model.W1, model.b1, model.W2, model.b2]))

        # 損失函數變化紀錄
        loss_history.append(loss.numpy())
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

    # 畫出損失曲線
    plt.plot(loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()

    return model

# 進行推理（預測）


def predict(model, data):
    logits = model.forward(data)
    return tf.sigmoid(logits).numpy() > 0.5  # 轉換成 0 或 1

# 計算模型準確率


def evaluate_accuracy(model, data, labels):
    predictions = predict(model, data)
    accuracy = np.mean(predictions == labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# 視覺化決策邊界
def plot_decision_boundary(model, data, labels):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_data = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # 預測網格點
    predictions = predict(model, grid_data).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, colors=['white', 'gray'], alpha=0.1)
    plt.scatter(class_0[:, 0], class_0[:, 1],
                alpha=0.7,
                linewidth=0.1,
                s=40,
                color='#003060',
                label='Class 0'
                )

    plt.scatter(class_1[:, 0], class_1[:, 1],
                alpha=0.3,
                linewidth=0.1,
                s=40,
                color='black',
                label='Class 1'
                )
    plt.legend()

    plt.title("Decision Boundary")
    plt.show()
# %%
model = MultilayerPerceptron(
    input_dim=2, hidden_dim=5)  # 建立模型，2 個輸入特徵，5 個隱藏層神經元
model = train_model(model, data, labels, learning_rate=0.1, epochs=100)  # 訓練模型
evaluate_accuracy(model, data, labels)  # 計算準確率
plot_decision_boundary(model, data, labels)  # 繪製決策邊界
# %%
