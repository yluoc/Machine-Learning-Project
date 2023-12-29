import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd


def ldaLearn(X,y):
    d = X.shape[1]
    unique_classes = np.unique(y)
    k = len(unique_classes)

    means = np.zeros((d, k))
    covmat = np.cov(X, rowvar=False)  

    for i in range(k):
        class_data = X[y.flatten() == unique_classes[i]]
        class_mean = np.mean(class_data, axis=0)
        means[:, i] = class_mean
    return means,covmat

def qdaLearn(X,y):
    unique_labels = np.unique(y)
    k = len(unique_labels)
    d = X.shape[1]

    means = np.zeros((d, k))
    covmats = []

    for i, label in enumerate(unique_labels):
        X_class = X[y.flatten() == label]

        class_mean = np.mean(X_class, axis=0)
        means[:, i] = class_mean

        class_covmat = np.cov(X_class, rowvar=False, bias=True)
        covmats.append(class_covmat)
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    N= Xtest.shape[0]
    k = means.shape[1]

    ypred = np.zeros((N, 1))

    for i in range(N):
        x = Xtest[i, :].reshape(-1, 1)
        discriminant_scores = np.zeros((k, 1))

        for j in range(k):
            diff = x - means[:, j].reshape(-1, 1)
            covmat_inv = np.linalg.inv(covmat)
            discriminant_scores[j] = -0.5 * np.dot(np.dot(diff.T, covmat_inv), diff)

        ypred[i] = np.argmax(discriminant_scores) + 1

    correct_predictions = (ypred == ytest).sum()
    acc = correct_predictions / N
    return acc,ypred

def qdaTest(means, covmats, Xtest, ytest):
    N = Xtest.shape[0]
    k = means.shape[1]

    ypred = np.zeros((N, 1))

    for i in range(N):
        x = Xtest[i, :].reshape(-1, 1)
        discriminant_scores = np.zeros((k, 1))

        for j in range(k):
            diff = x - means[:, j].reshape(-1, 1)
            covmat_inv = np.linalg.inv(covmats[j])  # Access the correct covariance matrix
            p = -0.5 * np.dot(np.dot(diff.T, covmat_inv), diff)
            det_covmat = np.linalg.det(covmats[j])
            discriminant_scores[j] = -0.5 * np.log(det_covmat) + p

        ypred[i] = np.argmax(discriminant_scores) + 1

    correct_predictions = (ypred == ytest).sum()
    acc = correct_predictions / N
    return acc, ypred

def learnOLERegression(X,y):
    X_transpose = np.transpose(X)
    w = np.dot(np.dot(np.linalg.inv(np.dot(X_transpose, X)), X_transpose), y)                              
    return w

def learnRidgeRegression(X,y,lambd):
    c = X.shape[1]                        
    XT = X.T

    lambdaI = lambd * np.identity(c)
    w = np.linalg.inv(XT.dot(X) + lambdaI).dot(XT).dot(y)

    return w  
                                        

def testOLERegression(w,Xtest,ytest):
    y_pred = np.dot(Xtest, w)  
    residual = ytest - y_pred
    mse = np.mean(residual ** 2)
    return mse

def regressionObjVal(w, X, y, lambd):
    w = w.reshape(-1, 1)
    
    error = (1/2 * ((y - X.dot(w)).T).dot(y - X.dot(w)) + 1/2 * (lambd * (w.T).dot(w)))
    
    error_grad = -(X.T).dot(y - X.dot(w)) + lambd * w

    error_grad = np.squeeze(np.asarray(error_grad))
    return error, error_grad

def mapNonLinear(x,p):
    Xp = np.ones((x.shape[0],p+1))
    
    for i in range(1,p+1):
        Xp[:,i] = np.power(x,i)

    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'), encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

ole_w = w_i.ravel().tolist()

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

lambdas = lambdas.ravel().tolist()
mses3 = mses3.ravel().tolist()
min_mse = min(mses3)
min_lambda = lambdas[mses3.index(min_mse)]
print("The optimal lambda: " + str(min_lambda))             # find the minimum mse in test data and find its corresponding lambda


ole_w = w_i.ravel().tolist()
ridge_w = w_l.ravel().tolist()
rel_ridge_weight = sum(i**2 for i in ridge_w)
rel_ole_weight = sum(i**2 for i in ole_w)
print("The relative magnitudes of weights learnt using ridge regression: " + str(rel_ridge_weight))
print("The relative magnitudes of weights learnt using OLE regression: " + str(rel_ole_weight))

# data = {'Lambda': lambdas, 'MSE_test': mses3}
# df = pd.DataFrame(data)
# print(df.to_string(index=False))

plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.
w_init = np.ones((X_i.shape[1], 1))
w_init = w_init.flatten()
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
#lambda_opt = lambdas[np.argmin(mses4_train)]
#print(lambda_opt)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

lambdas = lambdas.ravel().tolist()
mses4 = mses4.ravel().tolist()
min_mse4 = min(mses4)
min_lambda4 = lambdas[mses4.index(min_mse4)]
print("The optimal lambda for problem4: " + str(min_lambda4))

plt.show()

# Problem 5
pmax = 7
lambda_opt = 0.02 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
