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
df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].hist(
    bins=20, 
    figsize=(10, 8),
    color='gray',
    # xlabelsize=14,
    )
plt.tight_layout()
plt.show()

# %%
# 每個特徵的箱型圖
custom_palette = ["#D0D0D0", "#9D9D9D", "#6C6C6C", "#D0D0D0"]
plt.figure(figsize=(10, 8))
plt.title("Box plot", fontsize=16, pad=12)
sns.boxplot(color="gray",data=df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
plt.xticks(fontsize=14)
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


# %%
