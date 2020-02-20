import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


X = np.array([i*np.pi/180 for i in range(60,300,10)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*X + 7 + np.random.normal(0,3,len(X))

X = pd.DataFrame(X.reshape((len(X),1)))
y = pd.Series(np.array(y))
N = 300
P = 1


LR = LinearRegression(fit_intercept=True)
LR.fit_non_vectorised(X, y, 24, n_iter=500) 
past_thetas = LR.all_theta
past_costs = LR.allerror
LR.plot_line_fit(X,y,0,1)
LR.plot_surface(X,y,0,1)
LR.plot_contour(X, y, 0, 1)




