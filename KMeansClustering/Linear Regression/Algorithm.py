import numpy as np 
import pandas as pd
df= pd.read_csv('./MlLibrary/linear_regression_train.csv' )
df
x=df.iloc[:,1:26].to_numpy()

x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
y = df['Target'].to_numpy()
w = np.zeros(x.shape[1])
b = 0

def cost (x,y,w,b,lamda):
    m=x.shape[0]
    
    fwb=np.dot(x,w)+b
    err=fwb-y
    
    cost=np.sum(err**2)/(2*m)
    reg_cost=(lamda/(2*m))*np.sum(w**2)
    
    total_cost=cost+reg_cost
    return total_cost
def gradient (x,y,w,b,lamda):
    m, n = x.shape
    fwb=np.dot(x,w)+b
    err=fwb-y
    
    djdw=(np.dot(x.T,err)+lamda*w)/m
    djdb=np.sum(err)/m
    
    
    return djdw,djdb

def wb_calc(x,y,w,b,alpha,num_iters,lamda):
    for k in range(num_iters):
        djdw, djdb = gradient(x, y, w, b, lamda)
        # Update weights and bias
        w -= alpha * djdw
        b -= alpha * djdb
        
        
    return w,b


def predict(x, w, b):
    return np.dot(x, w.T) + b

alpha = 0.001
num_iters = 10000
lamda = 0
grad = gradient(x, y, w, b, lamda)
print(f"Initial gradient: djdw = {grad[0]}, djdb = {grad[1]}")

w, b = wb_calc(x, y, w, b, alpha, num_iters, lamda)
print(f"Optimized parameters: w = {w}, b = {b}")

y_predict = predict(x, w, b)

print("Predicted target values:")
print(y_predict)
a=cost(x,y,w,b,lamda)
print(a)
MSE=np.mean((y - y_predict)**2)
print(MSE)

y_mean = np.mean(y)
ss_res = np.sum((y - y_predict)**2)
ss_tot = np.sum((y- y_mean)**2)
r2_score = 1 - (ss_res / ss_tot)
print(f"R² Score: {r2_score}")


