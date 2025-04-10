# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# %%

# 生成 2D 資料點（兩類）
np.random.seed(40)  # 固定隨機種子
class_0 = np.random.randn(100, 2) + np.array([-2, 0])  # 第一類
class_1 = np.random.randn(100, 2) + np.array([2, 0])   # 第二類

print("前五筆資料", class_0[:5])

# 合併數據
X = np.vstack([class_0, class_1]) 
y = np.array([0] * 100 + [1] *
             100).astype(np.float32).reshape(-1, 1) 

print("資料向量維度x,y",X.shape, y.shape) 

# 合併資料
data = np.vstack((class_0, class_1)).astype(np.float32)

# 建立標籤
labels = np.array([0] * 100 + [1] * 100).astype(np.float32).reshape(-1, 1)
print("資料二元分類標籤", labels[:5]) 


# 繪製資料點
plt.scatter(class_0[:, 0], class_0[:, 1], color='purple', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='yellow', label='Class 1')
plt.legend()
plt.title("Generated Data")
plt.show()

# %%

# 定義神經網路
class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_dim):
        # 隱藏層權重
        self.W1 = tf.Variable(tf.random.normal([input_dim, hidden_dim]))
        self.b1 = tf.Variable(tf.zeros([hidden_dim])) 
        # 輸出層權重
        self.W2 = tf.Variable(tf.random.normal([hidden_dim, 1]))
        self.b2 = tf.Variable(tf.zeros([1])) 

    def forward(self, x):
        # 隱藏層 + ReLU 激活函數
        hidden_output = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)
        # 輸出層（線性變換）
        return tf.matmul(hidden_output, self.W2) + self.b2

# 定義交叉熵損失函數
def binary_crossentropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# 訓練函數
def train_model(model, data, labels, learning_rate=0.1, epochs=100):
    optimizer = tf.optimizers.SGD(learning_rate)

    loss_history = []
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            logits = model.forward(data)  # 前向傳播
            loss = binary_crossentropy_loss(labels, logits)  # 計算損失

        # 計算梯度並更新權重
        gradients = tape.gradient(loss, [model.W1, model.b1, model.W2, model.b2])  # 更新所有權重和偏置
        optimizer.apply_gradients(zip(gradients, [model.W1, model.b1, model.W2, model.b2]))

		# 損失函數變化紀錄
        loss_history.append(loss.numpy())
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

    # # 畫出損失曲線
    # plt.plot(loss_history)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Training Loss Curve")
    # plt.show()

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
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_data = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # 預測網格點
    predictions = predict(model, grid_data).reshape(xx.shape)

    plt.contourf(xx, yy, predictions, cmap="coolwarm", alpha=0.6)
    plt.scatter(data[:, 0], data[:, 1], c=labels[:, 0], cmap="coolwarm", edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

# 訓練與測試
# 假設你已經準備好了資料 data 和 labels
model = SimpleNeuralNetwork(input_dim=2, hidden_dim=5)  # 建立模型，2 個輸入特徵，5 個隱藏層神經元
model = train_model(model, data, labels, learning_rate=0.1, epochs=100)  # 訓練模型
evaluate_accuracy(model, data, labels)  # 計算準確率
plot_decision_boundary(model, data, labels)  # 繪製決策邊界