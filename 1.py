from keras.datasets import mnist 
import matplotlib.pyplot as plt

# 載入資料 mnist 數字
(train_feature, train_label),(test_feature, test_label) = mnist.load_data()
print("train_feature),(train_label): ",len(train_feature),len(train_label))
print("train_feature:",train_feature.shape,"train_label:",train_label.shape)

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)  # 設定畫布大小
    plt.imshow(image, cmap='binary') # 使用灰階
    plt.show()

show_image(train_feature[0])
print("train_label[0]: ",train_label[0])

def show_image_label_prediction(image, labels, predictions, start_id, num = 10):
    plt.gcf().set_size_inches(12, 14)
    if num>25: num = 25
    for i in range(num):
        ax = plt.subplot(5, 5, i+1)
        # 顯示黑白圖片
        ax.imshow(image[start_id], cmap='binary')

        # 有AI 預測結果資料, 才在標題顯示預測結果
        if(len(predictions) > 0):
            title += 'ai = ' + str(predictions[i])
            # 預測正確結果輸出
            title += (' (o)' if predictions[i]==labels[i] else ' (x)')
            title += '\nlabel =' + str(labels[i])
        # 沒有 AI 預測結果， 只在標題顯示真實數值
        else:
            title = 'label = ' + str(labels[i])

        # X, Y 軸不顯示刻度
        ax.set_title(title, fontsize = 12)
        ax.set_xticks([]);ax.set_yticks([])
        start_id += 1
    plt.show()

show_image_label_prediction(train_feature, train_label,[],0,10)

train_feature_vector = train_feature.reshape(len(train_feature),784).astype('float32')

test_feature_vector = test_feature.reshape(len(test_feature),784).astype('float32')

print('train_feature_vector: ',train_feature_vector.shape,'test_feature_vector: ',test_feature_vector.shape)
# print(train_feature_vector[0])

# 標準化 將255 轉換成 0 ~ 1
train_feature_narmalize = train_feature_vector / 255
test_feature_narmalize = test_feature_vector / 255
# print(train_feature_narmalize[0])

from keras.utils import np_utils
print(train_label[0:5]) # [5 0 4 1 9]
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)
print(train_label_onehot[0:5])

