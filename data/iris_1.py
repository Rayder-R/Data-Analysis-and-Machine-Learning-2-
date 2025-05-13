#%%
import pandas as pd
#%%
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv("../Dataset/iris/iris.data", header=None, names=columns)
#%%
# 顯示資料的行列數
print(df.shape)

# 顯示資料的第一行
print(df.head(1))

# 資料型態與欄位
print(df.info())
#%%

# 類別分布（class 欄位）
print(df["class"].value_counts())

#%%

# 數值欄位統計摘要（平均、標準差、最大最小值等）
print(df.describe())

