import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import cluster
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
k = 3

kmeans = cluster.KMeans(n_clusters=k, random_state=12)
kmeans.fit(X)

#查看data和欄位資訊
print(X)
#查看key值
print(iris.keys())
print("K-maens classification:")
print(kmeans.labels_)
#修正標籤錯誤位置
print("K-maens (Fix) classification:")
pred_y = np.choose(kmeans.labels_, [1,0,2]).astype(np.int64)
print(pred_y)
print("Real classification:")
print(y)

colmap = np.array(["r","g","y"])
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.subplots_adjust(hspace = .5)
plt.scatter(X["petal length (cm)"], X["petal width (cm)"],
            color = colmap[y])
plt.xlabel("Petal Lenght")
plt.ylabel("Petal Width")
plt.title("Real Classification")

plt.subplot(1,2,2)
plt.scatter(X["petal length (cm)"], X["petal width (cm)"],
            color = colmap[kmeans.labels_])           
plt.xlabel("Petal Lenght")
plt.ylabel("Petal Width")
plt.title("K-means Classification")
plt.show()

