
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical


import numpy as np
np.random.seed(10)

# 建立訓練資料和測試資料，包誇訓練資料、訓練標籤和測試特徵集、測試標籤
(train_feature, train_label), (test_feature, test_label) = mnist.load_data()

print(len(train_feature))

# 將 Feature 特徵值換為 60000*28*28*1 的 4 維矩陣
train_feature_vector = train_feature.reshape(
    len(train_feature), 28, 28, 1).astype('float32')
# prediction 測試用 test
test_feature_vector = test_feature.reshape(
    len(test_feature), 28, 28, 1).astype('float32')

# feature 特徵值標準化
train_feature_normalize = train_feature_vector / 255
# prediction 測試用 test
test_feature_normalize = test_feature_vector / 255

# label 轉換為 One-Hot Encoding 編碼
train_feature_onehot = to_categorical(train_label)
test_feature_onehot = to_categorical(test_label)

print(test_feature_onehot[1])
print(test_label[1])

# 建立模型
model = Sequential()
# 建立卷積層
model.add(Conv2D(filters=10,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape=(28, 28, 1),
                 activation='relu'))

# 建立池化層1
model.add(MaxPool2D(pool_size=(2, 2)))  # (10,14,14)

# 建立卷積層2
model.add(Conv2D(filters=20,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))

# 建立池化層2
model.add(MaxPool2D(pool_size=(2, 2)))  # (20,7,7)

# Dropout 層防止過度擬合，段開比例:0.2
model.add(Dropout(0.2))

# 建立平坦層:20*7*7=980個神經元
model.add(Flatten())

# 建立隱藏層
model.add(Dense(units=256, activation='relu'))

# 建立輸出層
model.add(Dense(units=10, activation='softmax'))

# 定義訓練方式
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# 以(train_feature_normalize, train_feature_onehot) 資料訓練，
# 訓練資料保留 20% 做驗證, 訓練 n 次, 每筆讀取 n 筆資料, 顯示簡易訓練過程
train_history = model.fit(x=train_feature_normalize,
                          y=train_feature_onehot, validation_split=0.2,
                          epochs=10, batch_size=200, verbose=2)

# 評估準確度
scores = model.evaluate(test_feature_normalize, test_feature_onehot)
print('\n準確率=', scores[1])

# save 模型
model.save('Mnist_cnn_model.h5')
print("\n Mnist_cnn_model.h5 模型儲存完畢")
model.save_weights("Mnist_cnn_model.weight")
print("\n Mnist_cnn_model.weight 模型參數儲存完畢")
