import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv("C:/Users/Harpreet Singh/Desktop/csv files/homeprics .csv")
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,marker='+',color='red')
reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)
reg.predict([[3300]])
reg.coef_
reg.intercept_
135.78767123*3300+180616.43835616432

plt.xlabel('area',fontsize=20)
plt.ylabel('price',fontsize=20)
plt.scatter(df.area,df.price,marker='+',color='red')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()

d=pd.read_csv("C:/Users/Harpreet Singh/Desktop/csv files/area - Sheet1.csv")
d.head(3)
p=reg.predict(d)
d['prices']=p
d
d.to_csv("prediction.csv", index='false')

