import matplotlib.pyplot as plt
# Keras
from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np

# 載入資料 mnist 
# 建立訓練資料和測試資料，包誇訓練特徵集、訓練標籤和測試特徵集、測試標籤
(train_feature, train_label),(test_feature, test_label) = mnist.load_data()

def show_image_label_prediction(image, labels, prediction, start_id, num = 10):
    plt.gcf().set_size_inches(12, 14)
    if num>10: num = 10
    for i in range(0, num):
        ax = plt.subplot(2, 5, 1+i)
        # 顯示黑白圖片
        ax.imshow(image[start_id], cmap='binary')

        # 有AI 預測結果資料, 才在標題顯示預測結果
        if(len(prediction) > 0):
            title = 'ai = ' + str(prediction[i])
            # 預測正確結果輸出
            title += (' (o)' if prediction[i]==labels[i] else ' (x)')
            title += '\nlabel =' + str(labels[i])
        # 沒有 AI 預測結果， 只在標題顯示真實數值
        else:
            title = 'label = ' + str(labels[i])

        # X, Y 軸不顯示刻度
        ax.set_title(title, fontsize = 10)
        ax.set_xticks([]);ax.set_yticks([])
        start_id += 1
    plt.show()

# show_image_label_prediction(train_feature, train_label,[],0,10)

# 將Features 特徵值換為784個 float 數字一為向量
test_feature_vector = test_feature.reshape(len(test_feature),784).astype('float32')

# 將Features 標準化 將255 轉換成 0 ~ 1
test_feature_narmalize = test_feature_vector / 255

print("載入模型 <Mnist_mlp_model.h5>")
model = load_model('Mnist_mlp_model.h5')


# 預測
prediction = np.argmax(model.predict(test_feature_narmalize), axis=1)

# 顯示圖像、預測值、真實值
show_image_label_prediction(test_feature, test_label, prediction, 0)