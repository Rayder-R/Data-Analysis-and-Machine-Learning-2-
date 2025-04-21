# %%
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

# %%
vocab_size = 10000   # 詞彙表大小
# %%
(train_data, train_labels), (test_data, test_labels) = keras.datasets.reuters.load_data(
    num_words=vocab_size
)

print("train_data shape:", train_data.shape,
      "train_labels shape:", train_labels.shape)
print("test_data shape:", test_data.shape,
      "test_labels shape:", test_labels.shape)
print("train_data[0]:", train_data[0])
print("train_labels[0]:", train_labels[0])


# %%
# Multi-hot encoding 函數
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


# 訓練與測試資料轉換為 Multi-hot encoding
train_data = vectorize_sequences(train_data, vocab_size)
test_data = vectorize_sequences(test_data, vocab_size)

# %%
print("train_data shape:", train_data.shape)
print("test_data shape:", test_data.shape)
print("train_data[0]:", train_data[0])


# %%
# Convert labels to one-hot
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

print("train_labels shape:", train_labels.shape)
print("test_labels shape:", test_labels.shape)
print("train_labels[0]:", train_labels[0])
print("test_labels[0]:", test_labels[0])

# %%
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10000,)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(46, activation='softmax')
])
model.build(input_shape=(None, vocab_size))  # (None, 10000)
model.summary()

# %%
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練模型
history = model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size=512,
    validation_split=0.2,
    # callbacks=[keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=3, restore_best_weights=True)]
)

# %%

# 繪製訓練與驗證損失
plt.figure(figsize=(12, 4))

# 損失圖
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 準確率圖
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %%
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {test_acc:.4f}')


# %%
