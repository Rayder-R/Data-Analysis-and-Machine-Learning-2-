import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical


def show_image_labels_prediction(images, label, prediction, start_id, num=10):
    plt.gcf().set_size_inches(12, 14)
    images = np.array(images)  # 確保是 NumPy 陣列
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[start_id].squeeze(), cmap="binary")

        if len(prediction) > 0:
            title = f"ai = {prediction[i]}"
            title += " (o)" if prediction[i] == label[i] else " (x)"
            title += f"\nlabel = {label[i]}"
        else:
            title = f"label = {label[i]}"

        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id += 1

    plt.ioff()
    plt.show(block=True)
    input("Press Enter to exit...")  # 讓視窗不會馬上關閉


# 建立 MNIST 資料
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()

# 將 Feature 特徵值換為 4 維矩陣
test_feature_vector = test_feature.reshape(
    len(test_feature), 28, 28, 1).astype("float32")

# 特徵值標準化
test_feature_normalize = test_feature_vector / 255

# Label 轉換為 One-Hot Encoding
test_label_onehot = to_categorical(test_label)

# 載入模型
model = "Mnist_cnn_model.h5"
print("載入模型: "+model)
model = load_model("Mnist_cnn_model.h5")

# 評估準確率
scores = model.evaluate(test_feature_normalize, test_label_onehot)
print("\n準確率=", scores[1])

# 預測
prediction = np.argmax(model.predict(test_feature_normalize), axis=1)

# 顯示結果
show_image_labels_prediction(test_feature, test_label, prediction, 0)
