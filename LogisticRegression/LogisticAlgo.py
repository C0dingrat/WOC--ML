import numpy as np
import pandas as pd
df=pd.read_csv('./MlLibrary/binary_classification_train.csv')
df

x=df.iloc[:, 1:21].to_numpy()
y=df['Class'].to_numpy()
x = (x- np.mean(x, axis=0)) / np.std(x, axis=0)

w=np.zeros([x.shape[1]])
b=0

def sig(z):
    
    g= 1/ (1+np.exp(-z))
    
    return g

def compute_cost(X, Y, w, b,lambda1):
    m = X.shape[0]
    z = np.dot(X, w) + b
    fwb = sig(z)
    cost = -(1/m) * np.sum(Y * np.log(fwb) + (1 - Y) * np.log(1 - fwb))
    reg_cost = (lambda1 / (2 * m)) * np.sum(w**2)
    return cost + reg_cost

def compute_gradient(X, Y, w, b, lambda1):
    m, n = X.shape
    z = np.dot(X, w) + b
    fwb = sig(z)
    error = fwb - Y
    djdw = (1/m) * np.dot(X.T, error) + (lambda1/m) * w
    djdb = (1/m) * np.sum(error)
    return djdw, djdb


def wb_calc(X, Y, w, b, alpha, num_iters, lambda1):
    
    
    for k in range(num_iters):
        djdw, djdb = compute_gradient(X, Y, w, b, lambda1)
        w -= alpha * djdw
        b -= alpha * djdb
        if k % 1000 == 0:
            cost = compute_cost(X, Y, w, b, lambda1)
            
            print(f"Iteration {k}: Cost {cost:.4f}")
    return w, b


alpha=0.001
num_iters=10000
lambda1=10^-3

gradient=compute_gradient(x,y,w,b,lambda1)
print(f"Final gradient: djdw = {gradient[0]}, djdb = {gradient[1]}")

w ,b= wb_calc(x,y,w,b,alpha,num_iters,lambda1)
print(f"Optimized parameters: w = {w}, b = {b}")



def predict(X, w, b):
    z = np.dot(X, w) + b
  
    return sig(z) >= 0.5



y_predict = predict(x, w, b)


print("Predicted target values:")
print(y_predict)

a=compute_cost(X_train,y_train,w,b,lambda1)
print(a)

correct_pred=0
for true, pred in zip(y,y_predict):
    
    if true==pred:
        correct_pred+=1
        


accuracy=correct_pred /len(y_train)
print(f"Accuracy: {accuracy*100:.2f}%")
