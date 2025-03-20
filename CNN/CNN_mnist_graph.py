import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 讀取 MNIST 手寫數字資料集
(train_feature, train_label), (test_feature, test_label) = keras.datasets.mnist.load_data()

# 將測試資料的特徵值調整為 4 維張量 (批次大小, 高度, 寬度, 通道數)
# 這是 CNN 的標準輸入格式，因為 MNIST 是灰階影像，所以通道數為 1
test_feature_vector = test_feature.reshape(
    len(test_feature), 28, 28, 1).astype("float32")

# 進行特徵標準化，將像素值 (0-255) 縮放至 0-1 之間
test_feature_normalize = test_feature_vector / 255

# 轉換標籤為 One-Hot 編碼，因為模型使用的是分類問題
test_label_onehot = keras.utils.to_categorical(test_label)

# 載入已訓練的 CNN 模型
model = "cnn_model.h5"
print("載入模型: "+model)
model = keras.models.load_model(model)

# 重新編譯模型，確保 metrics 設定正確
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 評估模型準確率，會輸出 loss 和 accuracy
scores = model.evaluate(test_feature_normalize, test_label_onehot)
print("\n準確率=", scores[1])

# 進行預測，並取出最大機率的類別索引
prediction = np.argmax(model.predict(test_feature_normalize), axis=1)

# 定義函式來顯示影像、真實標籤與 AI 預測結果
def show_image_labels_prediction(images, label, prediction, start_id, num=25):
    plt.gcf().set_size_inches(15, 15)
    images = np.array(images)  # 確保輸入為 NumPy 陣列
    if num > 100:  # 限制最多顯示 100 張圖片
        num = 100
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)  # 建立 5x5 子圖
        ax.imshow(images[start_id].squeeze(), cmap="binary")  # 顯示圖片（灰階）

        # 顯示預測結果與真實標籤
        if len(prediction) > 0:
            title = f"ai = {prediction[i]}"
            title += " (o)" if prediction[i] == label[i] else " (x)"  # 標註預測是否正確
            title += f"\nlabel = {label[i]}"
        else:
            title = f"label = {label[i]}"

        ax.set_title(title, fontsize=12)  # 設定標題
        ax.set_xticks([])  # 移除 x 軸刻度
        ax.set_yticks([])  # 移除 y 軸刻度
        start_id += 1  # 讀取下一張圖片

    # plt.ioff()  # 關閉互動模式，確保圖表正常顯示
    plt.subplots_adjust(hspace=2, wspace=2)  # Adjust space between subplots
    plt.show(block=True)  # 顯示圖片，並阻止視窗自動關閉
    input("Press Enter to exit...")  # 防止視窗立即關閉

# 顯示測試資料的預測結果
show_image_labels_prediction(test_feature, test_label, prediction, 0)
