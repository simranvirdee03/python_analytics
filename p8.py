import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
df=pd.read_csv("C:/Users/Harpreet Singh/Desktop/ml csv files/insurance_data.csv")

plt.scatter(df['age'],df['bought_insurance'],marker='+')

x_train,x_test,y_train,y_test=train_test_split(df[['age']],df['bought_insurance'],test_size=0.1)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
print(x_test)
lr=linear_model.LogisticRegression()
lr.fit(x_train,y_train)
print(lr.predict(x_test))
print(lr.score(x_test,y_test))
print(lr.predict_proba(x_test))
print(lr.predict([[25]]))