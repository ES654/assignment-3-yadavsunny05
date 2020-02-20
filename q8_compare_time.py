import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time





n_vals = [i for i in range(30,2000)]
p_vals = [i for i in range(10,1000)]

fin_intercept = False

P = 5
for N in n_vals:
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    startTime = time.time()
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_normal(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    endTime = time.time()
    emperical_n1.append(endTime-startTime)    


P = 5
for N in n_vals:
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    startTime = time.time()
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_vectorised(X, y,N) # here you can use fit_non_vectorised / fit_autograd methods
    endTime = time.time()
    emperical_n2.append(endTime-startTime)    
