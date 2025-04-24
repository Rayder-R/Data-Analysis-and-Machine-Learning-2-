from sklearn.metrics import log_loss
import numpy as np

# 真實標籤（One-Hot 編碼）
y_true = np.array([[1, 0, 0]])

# 模型預測的概率分佈
y_pred = np.array([[0.659, 0.242, 0.099]])

# 計算交叉熵損失
loss = log_loss(y_true, y_pred)
print(f'交叉熵損失: {loss:.3f}')



import numpy as np
from scipy.special import softmax

logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)
print(f'softmax : {loss:.3f}')
print(probabilities)