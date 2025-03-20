import tensorflow as tf
from tensorflow import keras

from keras.models import load_model

# 1. 載入 Keras H5 模型
model = load_model("Mnist_cnn_model.h5", compile=False)

# 2. 確保 TensorFlow 追蹤未追蹤的函數
concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
    tf.TensorSpec(model.input_shape, model.input.dtype)
)

# 3. 儲存為 SavedModel，確保它包含完整函數
tf.saved_model.save(model, "Mnist_saved_model", signatures=concrete_func)

print("模型成功轉換為 SavedModel！")
