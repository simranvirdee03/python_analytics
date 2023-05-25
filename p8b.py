import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model


df=pd.read_csv("C:/Users/Harpreet Singh/Desktop/ml csv files/HR_comma_sep.csv")

#Data Exploration and visualization
left_emp=df[df.left==1]
#print(left_emp.shape)
retained=df[df.left==0]
#print(retained.shape)

#Average number for all columns
#df.groupby('left').mean()

pd.crosstab(df.salary,df.left).plot(kind='bar')
pd.crosstab(df.Department,df.left).plot(kind='bar')

#we notice satisfaction level,avg_hrs,promotion and salary depends

subdf=df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
salary_dummy=pd.get_dummies(subdf.salary,prefix="salary")
df_med_dummy=pd.concat([subdf,salary_dummy],axis='columns')
X=df_med_dummy.drop('salary',axis='columns')
Y=df['left']
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.3)
lr=linear_model.LogisticRegression()
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))
print(lr.predict(x_test))
