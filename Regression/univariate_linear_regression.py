# %%
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
# %% 環境準備

# 設定中文字型
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # Windows

# 載入加州房價資料集
california = fetch_california_housing()
print(california.DESCR)  # 資料集的描述

# %% 轉換Pandas資料格式

# 將資料集轉換為 DataFrame 和 Series 物件
# X 為特徵資料，y 為目標變數
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name="Price")

# 資料集與目標變數的查看
print(y.describe())
X.head()
# %% 選擇 AveRooms 特徵進行單變量線性回歸

# 查看 AveRooms 特徵的分佈
print(X['AveRooms'].describe())

# 對 AveRooms 特徵評估誤差與視覺化
X_AveRooms = X[["AveRooms"]]

# 建立並訓練模型
model_uni = LinearRegression()
model_uni.fit(X_AveRooms, y)

# 預測
y_pred_uni = model_uni.predict(X_AveRooms)

# 評估誤差
mse_uni = mean_squared_error(y, y_pred_uni)
rmse_uni = np.sqrt(mse_uni)

print("評估誤差: ")
print(f"單變量模型 MSE: {mse_uni:.4f}")
print(f"單變量模型 RMSE: {rmse_uni:.4f}")
# %% AveRooms 特徵的視覺化
plt.figure(figsize=(8, 6))

plt.scatter(
    X["AveRooms"], y,
    color='lightgray',       # 內部填色：淡灰色或其他淺色
    linewidth=0.3,           # 外框線條寬度
    s=40,                    # 點的大小，預設是20，適度加大
    alpha=0.5,               # 透明度：讓點半透明但仍清晰
    label='實際資料'
)
plt.plot(X["AveRooms"], y_pred_uni, color='black',linewidth=2, linestyle='-', label='回歸線')
# 標籤與標題
plt.xlabel("平均房間數 (AveRooms)")
plt.ylabel("房價")
plt.title("單變量線性回歸")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.show()
# %% 選擇 MedInc 特徵進行單變量線性回歸

# 查看 MedInc 特徵的分佈
print(X['MedInc'].describe())

X_MedInc = X[["MedInc"]]

# 建立並訓練模型
model_uni = LinearRegression()
model_uni.fit(X_MedInc, y)

# 預測
y_pred_uni = model_uni.predict(X_MedInc)

# 評估誤差
mse_uni = mean_squared_error(y, y_pred_uni)
rmse_uni = np.sqrt(mse_uni)

print("評估誤差: ")
print(f"單變量模型 MSE: {mse_uni:.4f}")
print(f"單變量模型 RMSE: {rmse_uni:.4f}")

# %% MedInc 特徵的視覺化
plt.figure(figsize=(8, 6))
plt.scatter(X["MedInc"], y,
            color='lightgray',       # 內部填色：淡灰色或其他淺色
            linewidth=0.3,           # 外框線條寬度
            s=40,                    # 點的大小，預設是20，適度加大
            alpha=0.5,               # 透明度：讓點半透明但仍清晰
            label='實際資料'
            )
plt.plot(X["MedInc"], y_pred_uni, color='black', label='回歸線')
plt.xlabel("平均收入 (MedInc)")
plt.ylabel("房價")
plt.title("單變量線性回歸")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.show()
# %% 對 MedInc 特徵進行異常值處理

# 使用四分位距法（IQR Method）
Q1 = X["MedInc"].quantile(0.25)
Q3 = X["MedInc"].quantile(0.75)
IQR = Q3 - Q1
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")

# 過濾掉異常值 X
filtered_data = X[(X["MedInc"] >= (Q1 - 1.5 * IQR)) &
                  (X["MedInc"] <= (Q3 + 1.5 * IQR))]
# 過濾後的 y
filtered_y = y[filtered_data.index]

X_MedInc = filtered_data[["MedInc"]]

# 重新建立並訓練模型(過濾異常值版本)
model_uni = LinearRegression()
model_uni.fit(X_MedInc, filtered_y)

# 預測
y_pred_uni = model_uni.predict(X_MedInc)

# 評估誤差
mse_uni = mean_squared_error(filtered_y, y_pred_uni)
rmse_uni = np.sqrt(mse_uni)

print("評估誤差: ")
print(f"單變量模型 MSE: {mse_uni:.4f}")
print(f"單變量模型 RMSE: {rmse_uni:.4f}")

# %% 修正後的 MedInc 資料視覺化
plt.figure(figsize=(8, 6))
plt.scatter(filtered_data["MedInc"], filtered_y,
            color='lightgray',       # 內部填色：淡灰色或其他淺色
            linewidth=0.3,           # 外框線條寬度
            s=40,                    # 點的大小，預設是20，適度加大
            alpha=0.5,               # 透明度：讓點半透明但仍清晰
            label='實際資料'
            )
plt.plot(filtered_data["MedInc"], y_pred_uni, color='black', label='回歸線')
plt.xlabel("平均房間數 (MedInc)")
plt.ylabel("房價")
plt.title("單變量線性回歸")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.show()

# %% 單一數值預測範例
example_income = pd.DataFrame({"MedInc": [5.0]})
predicted_price = model_uni.predict(example_income)
print(f"當 MedInc = 5.0 時，預測房價為: {predicted_price[0]:.4f}")
