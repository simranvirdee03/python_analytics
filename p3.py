import pandas as pd
from sklearn import linear_model
import math
df=pd.read_csv("C:/Users/Harpreet Singh/Desktop/csv files/homeprices - Sheet1.csv")
med_b=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(med_b)
reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
print(reg.coef_)
print(reg.intercept_)
print("First Question",reg.predict([[3000,3,40]]))
print("Second Question",reg.predict([[2500,4,5]]))