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
    .to_csv(r'dtr_dataset.csv',index=False,header=True)


#reading csv file
df = pd.read_csv("dtr_dataset.csv",sep=",")

x = df.loc[:,["level"]].values.reshape(-1,1)
y = df.loc[:,["price"]].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

tree_reg.predict([[5.5]])
# arka tarafda information entropy matematiği kullanılır.

#visualization

# terminal leafler arası geçiş i 
# düzgün bir şekilde gösterebilmek için x_ oluşturuldu.s
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1);

plt.scatter(x,y,color="red")
plt.plot(x_,tree_reg.predict(x_),color="green")
plt.xlabel("level")
plt.ylabel("price")

