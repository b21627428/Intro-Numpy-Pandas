#import library
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

#creating csv file
cars = {
    "price":[60,70,80,100,120,150,200,250,300,400,500,750,1000,2000,3000],
    "speed":[180,180,200,200,200,220,240,240,300,350,350,360,365,365,365]
    }
pd.DataFrame(cars,columns=["price","speed"])\
    .to_csv(r'plr_dataset.csv',index=False,header=True)



#reading csv file
df = pd.read_csv("mlr_dataset.csv",sep=",")



