#import library
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import pandas as pd


#creating csv file
people = {
    "experience":[0.5,0,1,5,8,4,15,7,3,2,12,10,14,6],
    "salary":[2500,2250,2750,8000,9000,6900,20000,8500,6000,3500,15000,13000,18000,7500],
    }
pd.DataFrame(people,columns=["experience","salary"])\
    .to_csv(r'lr_dataset.csv',index=False,header=True)



#reading csv file
df = pd.read_csv("lr_dataset.csv",sep=",")


#plot data
plt.scatter(df["experience"],df["salary"])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression()

experience = df["experience"].values.reshape(-1,1)
salary = df["salary"].values.reshape(-1,1)

linear_reg.fit(experience,salary)

#prediction
b0 = linear_reg.predict([[0]])
bias = linear_reg.intercept_
bias == b0

b1 = linear_reg.coef_  


#visualization
import numpy as np

array = np.array([i for i in range(16)]).reshape(-1,1)
y_head = linear_reg.predict(array)

plt.scatter(experience,salary)
plt.plot(array,y_head,color="red")
















