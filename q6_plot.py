import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

fit_intercept=True

x = np.array([i*np.pi/180 for i in range(60,1300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

theta = []

LR = LinearRegression(True)
d = [1,3,5,7,9]
for j in range(300,700,100):
    theta1=[]
    degree=[]
    for i in [1,3,5,7,9]:
        poly = PolynomialFeatures(i,False)
        X=poly.transform(np.transpose([x]))
        LR = LinearRegression(fit_intercept=False)
        LR.fit_normal(pd.DataFrame(X[:j]),pd.Series(y[:j]))# here you can use fit_non_vectorised / fit_autograd methods
        theta1.append(np.linalg.norm(LR.coef_))
        degree.append(i)
        print(theta1)
    plt.scatter(degree,theta1,label="N = "+str(j))
    plt.legend(prop={'size': 6},borderpad=2)

plt.xlabel("degree")
plt.ylabel("theta")
plt.show()


