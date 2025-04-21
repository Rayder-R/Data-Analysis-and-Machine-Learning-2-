#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv("../Dataset/iris/iris.data", header=None, names=columns)

print("Pandas DataFrame 物件型態", type(df))

# %%
# 每個特徵的直方圖
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.show()

# %%
# 每個特徵的箱型圖
plt.figure(figsize=(10, 8))
sns.boxplot(data=df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
plt.show()

# %%
# 配對圖，視覺化特徵間的關係
sns.pairplot(df, hue="class", diag_kind="hist")
plt.show()

# %%
# 相關矩陣
correlation = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

