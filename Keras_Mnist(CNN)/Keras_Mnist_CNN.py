from operator import mod
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
np.random.seed(10)


def show_imasge_labels_prediction(images, label, prediction, star_id, num=10):
    plt.gcf().set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        # 顯示黑白圖片
        ax.imshow(images[star_id], cmap='binary')

        # 有 AI 預測結果資料 在標題顯示預測結果
        if(len(prediction) > 0):
            title = 'ai = ' + str(prediction[i])
            # 預測正確顯示 (o), 錯誤顯示 (x)
            title += (' (o)' if prediction[i] == label[i] else ' (x)')
            title += '\nlabel = ' + str(label[i])
            # 沒有 AI 預測結果資料, 只在標題顯示真實數值
        else:
            title = 'label = ' + str(label[i])
        ax.set_title(title,fontsize=12)
        ax.set_xtick([]);ax.set_ytick([])
        star_id += 1
    plt.show()

# 建立訓練資料和測試資料，包誇訓練資料、訓練標籤和測試特徵集、測試標籤
(train_feature, train_label),\
(test_feature, test_label) = mnist.load_data()

print(len(train_feature))
# 將 Feature 特徵值換為 60000*28*28*1 的 4 維矩陣
train_feature_vector = train_feature.reshape(len(train_feature),28,28,1).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature),28,28,1).astype('float32')

# feature 特徵值標準化
train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255
# print(train_feature_normalize )

# label 轉換為 One-Hot Encoding  編碼
train_feature_onehot = np_utils.to_categorical(train_label)
test_feature_onehot = np_utils.to_categorical(test_label)
print(test_feature_onehot[1])
print(test_label[1])

# 建立模型
model = Sequential()
# 建立卷積層
model.add(Conv2D(filter=10,
                kernel_size=(3,3),
                padding='same',
                input_shape=(28,28,1),
                activation='relu'))

# 建立池化層1
model.add(MaxPool2D(pool_size=(2,2))) #(10,14,14)

# 建立卷積層2
model.add(Conv2D(filters=20,
                kernel_size=(3,3),
                padding='same',
                activation='relu'))

# 建立池化層2
model.add(MaxPool2D(pool_size=(2,2))) #(20,7,7)

# Dropout 層防止過度擬合，段開比例:2.0
model.add(Dropout(0.2))

# 建立平坦層:20*7*7=980個神經元
model.add(Flatten())

# 建立隱藏層
model.add(Dense(units=256, activation='relu'))

# 建立輸出層
