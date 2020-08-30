#import library
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import pandas as pd
import numpy as np

#creating csv file
tickets = {
    "level":[1,2,3,4,5,6,7,8,9,10],
    "price":[100,80,70,60,50,40,30,20,10,5]
    }
pd.DataFrame(tickets,columns=["level","price"])\
    .to_csv(r'rfr_dataset.csv',index=False,header=True)


#reading csv file
df = pd.read_csv("rfr_dataset.csv",sep=",")

x = df["level"].values.reshape(-1,1)
y = df["price"].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor

# Popülasyondan her zaman aynı sample ı çekebilmek için
# 42 veya başka bir sayı veriyoruz.Id gibi düşünülebilir.
# Böylelikle gelecek sefer aynı şekilde sample çekecektir.
# Sample üzerinde 100 adet decision tree kullanılıp
# ortalaması alınır.
rf = RandomForestRegressor(n_estimators = 100,random_state=42)
rf.fit(x,y)

rf.predict([[5.5]])

#visualization

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

plt.scatter(x,y,color="red")
plt.plot(x_,rf.predict(x_),color="green")

#R_square değeri
# 1 e ne kadar yakınsa modelimiz o kadar iyidir.

from sklearn.metrics import r2_score

#Random Forest R_square
y_head = rf.predict(x) 
r2_score(y,y_head)

#Linear Regression R_square
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)
y_head = lr.predict(x)
r2_score(y,y_head)





