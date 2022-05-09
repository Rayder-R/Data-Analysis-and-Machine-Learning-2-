import matplotlib.pyplot as plt
# Keras
from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense

import random
import numpy as np
# 載入資料 mnist 
# 建立訓練資料和測試資料，包誇訓練特徵集、訓練標籤和測試特徵集、測試標籤
(train_feature, train_label),(test_feature, test_label) = mnist.load_data()
print("(train_feature),(train_label): ",len(train_feature),len(train_label))
print("train_feature:",train_feature.shape,"train_label:",train_label.shape)

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(3,3)  # 設定畫布大小
    plt.imshow(image, cmap='binary') # 使用灰階
    plt.show()

# show_image(train_feature[0])
print("train_label[0]: ",train_label[0])

def show_image_label_prediction(image, labels, prediction, num = 10):
    plt.gcf().set_size_inches(12, 14)
    if num>10: num = 10
    for i in range(0, num):
        num60000 = random.randrange(1, 10000)
        ax = plt.subplot(2, 5, 1+i)
        # 顯示黑白圖片
        ax.imshow(image[num60000], cmap='binary')

        # 有AI 預測結果資料, 才在標題顯示預測結果
        if(len(prediction) > 0):
            title = 'ai = ' + str(prediction[num60000])
            # 預測正確結果輸出
            title += (' (o)' if prediction[num60000]==labels[num60000] else ' (x)')
            title += '\nlabel =' + str(labels[num60000])
        # 沒有 AI 預測結果， 只在標題顯示真實數值
        else:
            title = 'label = ' + str(labels[num60000])

        # X, Y 軸不顯示刻度
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([]);ax.set_yticks([])
    plt.show()

# show_image_label_prediction(train_feature, train_label,[],0,10)

# 將Features 特徵值換為784個 float 數字一為向量
train_feature_vector = train_feature.reshape(len(train_feature),784).astype('float32')

test_feature_vector = test_feature.reshape(len(test_feature),784).astype('float32')

print('train_feature_vector: ',train_feature_vector.shape,'test_feature_vector: ',test_feature_vector.shape)

# 將Features 標準化 將255 轉換成 0 ~ 1
train_feature_narmalize = train_feature_vector / 255

test_feature_narmalize = test_feature_vector / 255

# print("train_feature_narmalize[0]:",train_feature_narmalize[0])

# label 轉換為 One-Hot Encoding 編碼 
train_label_onehot = np_utils.to_categorical(train_label)

test_label_onehot = np_utils.to_categorical(test_label)

print("train_label[0:5]:",train_label[0:5]) # [5 0 4 1 9]
print("train_label_onehot[0:5]:\n",train_label_onehot[0:5])

# 建立模型
model = Sequential()
# 建立輸出層和隱藏層, 輸入層:784, 隱藏層:256, 輸出層:10
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))
# 建立輸出層
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 訓練模型，訓練方式
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(x = 特徵值, y = 標籤, validation_split = 驗證資料百分比, 
#           epochs = 訓練次數, batch_size = 每批次有多少筆 verbose = n)
train_history = model.fit(x=train_feature_narmalize, y=train_label_onehot, \
                          validation_split=0.2, epochs=20, batch_size=200, verbose=2)

# 評估準確率
scores = model.evaluate(test_feature_narmalize, test_label_onehot)
print("\n準確率=", scores[1])

# 預測
prediction = np.argmax(model.predict(test_feature_narmalize), axis=1)

# 顯示圖像、預測值、真實值
show_image_label_prediction(test_feature, test_label, prediction)

model.save('.\Keras_Mnist(MLP)\Mnist_mlp_model3.h5')
print("模型儲存 <Mnist_mlp_model.h5>")