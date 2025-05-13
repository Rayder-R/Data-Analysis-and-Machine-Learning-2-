# %%

import tensorflow as tf
import keras as keras
import numpy as np

import matplotlib.pyplot as plt
# %%


# %%

# 標準化數據
X = np.array([[100, 150, 130, 160, 200, 250, 300] * 4], dtype=np.float32).reshape(1, 28, 1) / 310
y = np.array([[110, 160, 140, 170, 210, 260, 310] * 4], dtype=np.float32).reshape(1, 28, 1) / 310

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, input_shape=(28, 1), activation='tanh', return_sequences=True),
    keras.layers.Dense(1)
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
# 訓練模型並記錄歷史
history = model.fit(X, y, epochs=100, verbose=0)

# 繪製 loss 曲線
plt.plot(history.history['loss'], label='Loss')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# %%


# 預測結果
pred = model.predict(X)
print("Predicted next month income:", pred[0] * 310 )
print(pred.shape)

# %%
