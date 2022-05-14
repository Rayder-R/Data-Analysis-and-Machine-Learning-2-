import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import load_model

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
# train_feature_vector = train_feature.reshape(len(train_feature),28,28,1).astype('float32')

# prediction 測試用 test
test_feature_vector = test_feature.reshape(len(test_feature),28,28,1).astype('float32')

# feature 特徵值標準化
# train_feature_normalize = train_feature_vector/255

# prediction 測試用 test
test_feature_normalize = test_feature_vector/255


print("載入模型 Mnist_cnn_model.h5")
model = load_model('.\Keras_Mnist(CNN)\Mnist_cnn_model.h5')

# 預測  (函數版本已不是用)
# prediction = model.predict_classes(test_feature_normalize) 已棄用 classes .astype(int)
prediction = model.predict(test_feature_normalize)
print(prediction)
prediction = np.argmax(model.predict(test_feature_normalize), axis=1)
# prediction = model.predict(test_feature_normalize).astype(int)
# prediction = np.argmax(model.predict(test_feature_normalize), axis=1)

# 顯示
show_imasge_labels_prediction(test_feature,test_label,prediction,0)