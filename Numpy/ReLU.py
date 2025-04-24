import numpy as np

def relu(x):
    return np.maximum(0, x)

# 測試 ReLU
x = np.array([-2, -1, 0, 1, 2])
y = relu(x)
print(y)  # 輸出: [0 0 0 1 2]


import matplotlib.pyplot as plt

# 生成輸入數據
x = np.linspace(-5, 5, 100)
y = relu(x)

# 繪製圖表
plt.plot(x, y, label="ReLU", color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("ReLU Activation Function")
plt.xlabel("Input (x)")
plt.ylabel("Output (ReLU(x))")
plt.legend()
plt.grid()
plt.show()
