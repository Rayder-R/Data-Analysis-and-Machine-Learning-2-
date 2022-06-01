from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# https://towardsdatascience.com/end-to-end-machine-learning-project-telco-customer-churn-90744a8df97d

df = pd.read_csv('CH-16 ML_分群\Telco-Customer-Churn.csv')
# df.Dataframe(columns= ['customerID','gender','tenure','MonthlyCharges','TotalCharges'])
# (data= np.c_[iris['data'], iris['target']],
#                      columns= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])

df.head()
print(df.head())
C_df = pd.DataFrame(columns=['顧客ID','性別',''])
print(df['customerID'])
# df.columns('customerID')
# print(df.head())