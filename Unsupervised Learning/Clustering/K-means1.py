import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
Arr_Length = np.array([51, 46, 51, 45, 51, 50, 33,38, 37, 33, 33, 21, 23, 24])
Arr_weight = np.array([10.2, 8.8, 8.1, 7.7, 9.8, 7.2, 4.8,4.6, 3.5, 3.3, 4.3, 2.0, 1.0, 2.0])
df = pd.DataFrame({
         "Length":[51, 46, 51, 45, 51, 50, 33,
                   38, 37, 33, 33, 21, 23, 24],
         "weight":[10.2, 8.8, 8.1, 7.7, 9.8, 7.2, 4.8,
                   4.6, 3.5, 3.3, 4.3, 2.0, 1.0, 2.0]}) 

k = 3

kmean = cluster.KMeans(n_clusters=k, random_state=12)
kmean.fit(df)
print(Arr_Length)
print(Arr_weight)
print(kmean.labels_)

colmap = np.array(["r","g","y"])
plt.scatter(df["Length"],df["weight"], color=colmap[kmean.labels_])
plt.show()