import pandas as pd
from sklearn import linear_model
import pickle
df=pd.read_csv("C:/Users/Harpreet Singh/Desktop/ml csv files/HP - Sheet1.csv")
print(df)
reg=linear_model.LinearRegression()
reg.fit(df[['Year']],df['Salary'])
a=reg.predict([[2030]])
print(a)
with open('model_pickle','wb') as f:
    pickle.dump(reg,f)
with open('model_pickle','rb') as f:
    mp=pickle.load(f)
mp.predict([[2030]])