#import library
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

#creating csv file
people = {
    "experience":[0.5,0,1,5,8,4,15,7,3,2,12,10,14,6],
    "salary":[2500,2250,2750,8000,9000,6900,20000,8500,6000,3500,15000,13000,18000,7500],
    "age": [22,21,23,25,28,23,35,29,22,23,32,30,34,27]
    }
pd.DataFrame(people,columns=["experience","salary","age"])\
    .to_csv(r'mlr_dataset.csv',index=False,header=True)



#reading csv file
df = pd.read_csv("mlr_dataset.csv",sep=",")

x = df.loc[:,["experience","age"]].values
y = df["salary"].values.reshape(-1,1)

multiple_linear_reg = LinearRegression()

multiple_linear_reg.fit(x,y)

b0 = multiple_linear_reg.predict([[0,0]])
coefficents = multiple_linear_reg.coef_

# experience = 10 , age = 35 vs experince = 5 ,age =35 
multiple_linear_reg.predict(np.array([[10,35],[5,35]])) 
