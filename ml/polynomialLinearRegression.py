#import library
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import pandas as pd
from sklearn.linear_model import LinearRegression

#creating csv file
cars = {
    "price":[60,70,80,100,120,150,200,250,300,400,500,750,1000,2000,3000],
    "speed":[180,180,200,200,200,220,240,240,300,350,350,360,365,365,365]
    }
pd.DataFrame(cars,columns=["price","speed"])\
    .to_csv(r'plr_dataset.csv',index=False,header=True)



#reading csv file
df = pd.read_csv("plr_dataset.csv",sep=",")

x = df["price"].values.reshape(-1,1)
y = df["speed"].values.reshape(-1,1)

lr = LinearRegression()

# bu dataset ine linear regresyon un neden uygun olmadığını gösterir.
lr.fit(x,y)
plt.scatter(x,y,color="red")
plt.plot(x,lr.predict(x))

#Polynomial Linear Regression
from sklearn.preprocessing import PolynomialFeatures
polynomial_reg = PolynomialFeatures(degree = 5)

x_polynomial = polynomial_reg.fit_transform(x)
lr.fit(x_polynomial,y)

#visualization
plt.scatter(x,y,color="red")
plt.plot(x,lr.predict(x_polynomial))



