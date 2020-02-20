import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import HTML, Image


class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.X = None
        self.y = None
        self.error = None
        self.allerror = None
        self.all_theta = None


        pass

    def fit_non_vectorised(self, X, y, batch_size, n_iter=500, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        if(self.fit_intercept):
            X = np.concatenate([np.ones((len(y), 1)), np.array(X)], axis=1)
        X = np.array(X)
        y = np.array(y)
        self.X = X
        self.y =y
        self.coef_ = [0 for i in range(X.shape[1])]
        self.error = [0 for i in range(X.shape[1])]
        self.all_theta = [0 for i in range(n_iter)]
        temp = []
        self.allerror = []
        for iteration in range(n_iter):
            for i in range(0,X.shape[0],batch_size):
                temp_residual = []
                for j in range(i,min(i+batch_size,X.shape[0])):
                    temp_y = 0
                    for columns in range(X.shape[1]):
                        temp_y+= self.coef_[columns]*X[j][columns]
                    temp_residual.append(y[j] - temp_y)
                for columns in range((X.shape[1])):
                    descent = 0
                    for j in range(i,min(i+batch_size,X.shape[0])):
                        descent -= temp_residual[j]*X[j][columns]
                    if(lr_type == "constant"):
                        self.coef_[columns] -= lr*(descent/min(i+batch_size,X.shape[0]))
                    else:
                        self.coef_[columns] -= (lr/iteration)*(descent/min(i+batch_size,X.shape[0]))  
            h = np.dot(X, self.coef_)
            self.error = np.sum((h-y)**2)/len(y)
            self.allerror.append(self.error)
            t1 = self.coef_
            self.all_theta[iteration] = t1.copy()
        
        pass

    def fit_vectorised(self, X, y,batch_size, n_iter=500, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        
        if(self.fit_intercept):
            X = np.concatenate([np.ones((len(y), 1)), np.array(X)], axis=1)
        self.coef_ = [0 for i in range(X.shape[1])]
        self.X = X
        self.y =y

        for _ in range(n_iter):
            for i in range(0,X.shape[0],batch_size):
                X1 = np.array(X[i:min(i+batch_size,X.shape[0])])
                Y1 = np.array(y[i:min(i+batch_size,X.shape[0])])
                descent = np.matmul(np.transpose(X1), (np.matmul(X1,self.coef_)-Y1))
                if(lr_type == "constant"):
                    self.coef_ -= lr*(descent/Y1.size)
                else:
                    self.coef_ -= (lr/_)*(descent/Y1.size)
        return
        pass

    def fit_autograd(self, X, y, batch_size, n_iter=300, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        
        if(self.fit_intercept):
            X = np.concatenate([np.ones((len(y), 1)), np.array(X)], axis=1)
        self.X = X
        self.y =y
        def rmse(coeffs):
            return np.sum(np.square(np.dot(X1,coeffs)-Y1))/Y1.size
        gradient = grad(rmse)
        self.coef_ = np.array([0.0 for i in range(X.shape[1])])
        for _ in range(n_iter):
            for i in range(0,X.shape[0],batch_size):
                temp_residual = []
                X1 = np.array(X[i:min(i+batch_size,X.shape[0])])
                Y1 = np.array(y[i:min(i+batch_size,X.shape[0])])
                if(lr_type == "constant"):  
                    self.coef_-= lr*(gradient(self.coef_))
                else:
                    self.coef_ -= (lr/_)*(gradient(self.coef_))
        return
        pass

    def fit_normal(self, X, y):
        if(self.fit_intercept):
            X = np.concatenate([np.ones((len(y), 1)), np.array(X)], axis=1)
        X = np.array(X)
        y = np.array(y)
        self.y = y
        self.X = X
        self.coef_ = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),y))
        return
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''

        pass

    def predict(self, X):
        X = np.array(X)
        if(self.fit_intercept):
            X = np.concatenate([np.ones((len(self.y), 1)), X], axis=1) 
        return(X.dot(self.coef_))
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        pass

    def plot_surface(self, X, y, t_0, t_1):
        
        past_thetas = self.all_theta
        past_costs = self.allerror
        def calculate_cost(X, y, theta):
            n_samples = len(y)
            h = np.dot(X, theta)
            return np.sum((h-y)**2)/n_samples


        plt.style.use('seaborn-white')


        l = 25
        T1, T2 = np.meshgrid(np.linspace(1,6,200),np.linspace(0,10,200))
        X_ones = np.ones((len(y),1))
        X_bar = np.append(X_ones,X,axis=1)

        zs = np.array([calculate_cost(X_bar, y, np.array([t1,t2])) for t1, t2 in zip(np.ravel(T1), np.ravel(T2))])
        Z = zs.reshape(T1.shape)

        fig1, ax1 = plt.subplots(figsize = (7,7))
        fig2 = plt.figure(figsize = (7,7))
        ax2 = Axes3D(fig2)

        ax2.plot_surface(T1, T2, Z, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
        ax2.set_xlabel('theta 1')
        ax2.set_ylabel('theta 2')
        ax2.set_zlabel('error')
        ax2.set_title('RSS gradient descent')
        ax2.view_init(45, -45)

        line, = ax2.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
        point, = ax2.plot([], [], [], '*', color = 'red')
        display_value = ax2.text(2., 2., 27.5, '', transform=ax1.transAxes)

        def init_2():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            display_value.set_text('')
            return line, point, display_value


        past_thetas = np.array(past_thetas)

        def animate_2(i):
            print(i)
            # Animate line
            line.set_data(past_thetas[:i+1][:,t_0], past_thetas[:i+1][:,t_1])
            line.set_3d_properties(past_costs[i+1])
            # Animate points
            point.set_data(past_thetas[i+1][t_0], past_thetas[i+1][t_1])
            point.set_3d_properties(past_costs[i+1])
            # Animate display value
            display_value.set_text('Cost = ' + str(past_costs[i+1]))
            return line, point, display_value

        ax2.legend(loc = 1)

        anim2 = animation.FuncAnimation(fig2, animate_2, init_func=init_2,
                                    frames=len(past_costs)-1, interval=120, 
                                    repeat_delay=60, blit=True)
        anim2.save('animation2.gif', writer='imagemagick', fps = 15)
        print("check for animation2.gif at the workin Directory")
        return(plt)
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        



        pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        
        print(';dad')
        def init():
            line.set_data([], [])
            annotation.set_text('')
            return line, annotation

        def animate(i):
            x = np.linspace(-5, 20, 1000)
            y = past_thetas[i][t_1]*x + past_thetas[i][t_0] 
            line.set_data(x, y)
            annotation.set_text('Cost = %.2f e10' % (past_costs[i]/10000000000))
            return line, annotation

        past_thetas = self.all_theta
        past_costs = self.allerror
        fig, ax = plt.subplots()
        plt.title('Line Fit')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(X, y, color='red')
        line, = ax.plot([], [], lw=2)
        annotation = ax.text(-1, 700000, '')
        annotation.set_animated(True)
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=60, interval=0, blit=True)

        anim.save('animation.gif', writer='imagemagick', fps = 15)
        print("check for animation2.gif at the workin Directory")
        return(plt)
        pass

    def plot_contour(self, X, y, t_0, t_1):
        
        def calculate_cost(X, y, theta):
            n_samples = len(y)
            h = np.dot(X, theta)
            return np.sum((h-y)**2)/n_samples
        l = 25
        T1, T2 = np.meshgrid(np.linspace(1,6,200),np.linspace(0,10,200))
        X_ones = np.ones((len(y),1))
        X_bar = np.append(X_ones,X,axis=1)
        zs = np.array([calculate_cost(X_bar, y, np.array([t1,t2])) for t1, t2 in zip(np.ravel(T1), np.ravel(T2))])
        Z = zs.reshape(T1.shape)
        past_thetas = np.array(self.all_theta)
        past_costs = np.array(self.allerror)
        fig1, ax1 = plt.subplots(figsize = (7,7))
        ax1.contour(T1, T2, Z, 100, cmap='jet' )
        line, = ax1.plot([], [], 'r', label = 'Gradient descent', lw = 1.5)
        point, = ax1.plot([], [], '*', color = 'red', markersize = 4)
        value_display = ax1.text(0.02, 0.02, '', transform=ax1.transAxes)

        def init_3():
            line.set_data([], [])
            point.set_data([], [])
            value_display.set_text('')
            return line, point, value_display

        def animate_3(i):
            line.set_data(past_thetas[:i, t_0], past_thetas[:i, t_1])
            point.set_data(past_thetas[i, t_0], past_thetas[i, t_1])
            value_display.set_text('Min = ' + str(past_costs[i]))
            return line, point, value_display

        ax1.set_xlabel('theta 1')
        ax1.set_ylabel('theta 2')

        ax1.legend(loc = 1)

        anim3 = animation.FuncAnimation(fig1, animate_3, init_func=init_3,
                                    frames=len(past_thetas), interval=100, 
                                    repeat_delay=60, blit=True)
        anim3.save('animation3.gif', writer='imagemagick', fps = 15)
        print("check animation3.gif in thw working directory")
        return(plt)
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        pass
