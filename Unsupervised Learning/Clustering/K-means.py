# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plt.rcParams['font.family'] = 'Microsoft JhengHei'  # Windows 常用字型：微軟正黑體
# %%
# 讀取資料集
df = pd.read_csv('Wholesale customers data.csv')

# 顯示資料概況
print("資料欄位資訊:")
print(df.info())
print(df.shape)
# print("敘述統計摘要:")
# print(df.describe())

# 移除不重要欄位，從第2欄位開始
numeric_features = df.iloc[:, 2:]

# 特徵標準化（z-score 標準化）
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_features)

print("第一筆資料")
print(df.head(1))
print("標準化後的資料：")
print(scaled_data[:1])  # 顯示前5筆標準化後的資料

# %% 肘部法則（Elbow Method）找最佳 k 值
inertia_list = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia_list.append(kmeans.inertia_)

print("Inertia List:", inertia_list)
# 繪製
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia_list, 'bo-')
plt.xlabel('群數 k')
plt.ylabel('Inertia (總內部平方誤差)')
plt.title('Elbow Method — 判斷最佳群數')
plt.grid(True)
plt.show()

# %% 預測結果
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# 將分群結果加入原始資料框
df['Cluster'] = cluster_labels
print("分群結果：")
print(df['Cluster'].value_counts())


# 按 Cluster 分組計算每群的平均特徵值
group_profile = df.groupby('Cluster')[df.columns[2:8]].mean()

print("各群的平均特徵值：")
styled_profile = group_profile.style.set_caption("各群的平均特徵值") \
    .background_gradient(cmap='gray') \
    .format(precision=2)
styled_profile


# %%
# PCA 將資料降為 2 維以便視覺化
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# %% 繪製分群結果
plt.figure(figsize=(8, 6))
colors = ['#4F4F4F', '#003060', '#C4C4C4']

for cluster_id in range(k):
    subset = df[df['Cluster'] == cluster_id]
    plt.scatter(subset['PCA1'], subset['PCA2'],
                label=f'Cluster {cluster_id}', color=colors[cluster_id])

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('K-Means')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()


# %%
