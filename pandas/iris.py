#%%
import pandas as pd
#%%
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv("../Dataset/iris/iris.data", header=None, names=columns)
#%%
# 顯示資料的第一行
print(df.head(1))


# 資料型態與欄位
print(df.info())
#%%

# 類別分布（class 欄位）
print(df["class"].value_counts())

#%%

# 其他詳細

# 資料行列數（資料筆數與欄位數）
print(f"資料行列數: {df.shape}")

# 每欄資料型態（數值 or 類別）
print(df.dtypes)

# 空值檢查（確認是否有遺漏值）
print(df.isnull().sum())

# 數值欄位統計摘要（平均、標準差、最大最小值等）
print(df.describe())

# 再次列出統計摘要（與上面相同，可刪可留）
print(df.describe())



# %%
