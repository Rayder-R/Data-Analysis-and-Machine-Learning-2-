from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import sklearn.metrics as sm

k = 2
iris = datasets.load_iris()

iris_data = iris.data
iris_label = iris.target

Train_data, Test_data, Train_label, Test_label = train_test_split(iris_data, iris_label, test_size=0.5)

knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(iris_data,iris_label )
print(knn.predict(Test_data))
print(Test_label)
print(sm.accuracy_score(knn.predict(Test_data),Test_label))
# print(iris_data)
# print(iris_label)