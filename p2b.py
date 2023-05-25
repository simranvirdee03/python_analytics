import pandas as pd
from sklearn import linear_model
import pickle
import joblib
df = pd.read_csv("C:/Users/Harpreet Singh/Desktop/ml csv files/Canada.csv")
lm = linear_model.LinearRegression()
lm.fit(df[['year']], df.per_capita_income_US)
print("Prediction for 2020 = ", lm.predict([[2020]]))
with open('model_one','wb') as f:
    pickle.dump(lm,f)
with open('model_one','rb') as f:
    pp=pickle.load(f)
print(pp.predict([[2020]]))

joblib.dump(lm,'Model_02')
mo=joblib.load('Model_02')
print(mo.predict([[2020]]))

