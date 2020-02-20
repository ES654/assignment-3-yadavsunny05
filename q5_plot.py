import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
import pandas as pd

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

theta = []

LR = LinearRegression(True)
for i in range(2,20):
    poly = PolynomialFeatures(i, False)
    X = poly.transform(np.transpose([x]))
    LR.fit_normal(pd.DataFrame(data=X), pd.Series(y))
    theta.append(np.linalg.norm(LR.coef_))

plt.bar(x=[i for i in range(2,20)], height=theta)
plt.title("Normal Form")
plt.xlabel("Degree")
plt.ylabel("Theta")
plt.show()


