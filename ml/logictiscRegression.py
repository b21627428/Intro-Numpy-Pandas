import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

# Gereksiz columnları siliyoruz
data.drop(["Unnamed: 32","id"],axis=1,inplace=True)

# Object type ı int yapısına dönüştürüyoruz.
data["diagnosis"] = [ 1 if i == "M" else 0 for i in data["diagnosis"].values ]

cancerBool = data["diagnosis"].values
features = data.drop(["diagnosis"],axis=1)

#normalization
#tüm feature ları 0 ile 1 arasına scale ediyoruz ki , bütün feature lar dahil edilebilsin.
features = (features - np.min(features))/(np.max(features)-np.min(features))

#Train - Test splitting
from sklearn.model_selection import train_test_split

featuresTrain , \
    featuresTest , \
        cancerBoolTrain , \
            cancerBoolTest = \
                train_test_split(features,cancerBool,test_size=0.2,random_state=42)
featuresTrain = featuresTrain.T
featuresTest = featuresTest.T
cancerBoolTrain = cancerBoolTrain.T
cancerBoolTest = cancerBoolTest.T

#parameter initialize ve sigmoid function
#dimension = 30
def initializeWeightsAndBias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

#forward and backward propagation    
def forwardAndBackwardPropagation(w,b,x_train,y_train):
    #forward
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss =  -y_train * np.log(y_head) -(1-y_train) * np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    #backward
    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    
    return cost,gradients

#update weight and bias
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []

    #updating and learning parameters is number_of_iteration_times
    for i in range(number_of_iteration):
        #make forward and backward propagation and find cost and gradients
        cost,gradients = forwardAndBackwardPropagation(w, b, x_train, y_train)
        cost_list.append(cost)
        #lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration {}: {}".format(i,cost))
            
    #we learned parameters weight and bias
    parameters = {"weight":w,"bias":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation="vertical")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters,gradients,cost_list

def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
            
    return Y_prediction

def logisticRegression(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration):
    
    dimension = x_train.shape[0]
    w,b = initializeWeightsAndBias(dimension)
    
    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,number_of_iteration)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("test accuracy:{}".format(100 - np.mean(np.abs(y_prediction_test - y_test))*100))


logisticRegression(featuresTrain, cancerBoolTrain, featuresTest, cancerBoolTest, 1, 1000)


# logistic regression with sklearn
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(featuresTrain.T,cancerBoolTrain.T)
print("test accuracy:{}".format(lr.score(featuresTest.T,cancerBoolTest.T)))





        



