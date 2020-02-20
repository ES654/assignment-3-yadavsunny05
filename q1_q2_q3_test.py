
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *


np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

    
print('Normal')
for fit_intercept in [True,False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_normal(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
    print(LR.coef_)


print("vectorised")
for fit_intercept in [True,False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_vectorised(X, y,N) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
    print(LR.coef_)

print('Autograd')
for fit_intercept in [True,False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_autograd(X, y,N) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
    print(LR.coef_)


print("non vectorised")

for fit_intercept in [True,False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_non_vectorised(X, y,N) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))   
    print(LR.coef_)
