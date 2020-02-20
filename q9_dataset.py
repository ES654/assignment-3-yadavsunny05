import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression


np.random.seed(42)

ar1 = [i*np.pi/180 for i in range(60,300,10)]
ar2 = [i*np.pi/180 for i in range(30,200,5)][:len(ar1)]

ar1 = np.array(ar1)
ar2 = np.array(ar2)
ar3 = np.array(ar1 + ar2)

y = 4*ar1 + 5*ar2 + np.random.normal(0,3,len(ar1))


X = pd.DataFrame(ar1)
X[1] = ar2
X[2] = ar3

LR = LinearRegression(True)
LR.fit_vectorised(pd.DataFrame(data=X), pd.Series(y),batch_size = 5)

plt.bar(x=[0,1,2,3], height=LR.coef_)
plt.title("Multicolinear")
plt.xlabel("Theta")
plt.ylabel("Value")
plt.show()


