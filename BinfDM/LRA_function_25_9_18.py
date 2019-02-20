import os, sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
# ------------------------------------------
# Author:JP ; Adapted from Dr. Xiuxia Du's code
# Last Edit: 25/9/18
#
# Info: Does Linear Regression Analysis from input to class
#     Will also read in data from other sources for class obj instantiation
# ------------------------------------------

class do_LinRegAnal():
# Main constructor, multiple args: 
# 1) no argument
# 2) data from some data type source
# 3) user def sample number for random gen., beta-not, beta-1, and noise sigma            
    def d_lm(self, x, y, confidence=0.95):
        n = len(x)
        x_bar = np.mean(x)
        y_bar = np.mean(y)
        S_yx = np.sum((y - y_bar) * (x - x_bar))
        S_xx = np.sum((x - x_bar)**2)
# ====== estimate beta_0 and beta_1 ======
        beta_1_hat = S_yx / S_xx # also equal to (np.cov(x, y))[0, 1] / np.var(x)
        beta_0_hat = y_bar - beta_1_hat * x_bar
# ====== estimate sigma ======
# residual
        y_hat = beta_0_hat + beta_1_hat * x
        r = y - y_hat
        sigma_hat = np.sqrt(sum(r**2) / (n-2))
# ====== estimate sum of squares ======
# total sum of squares
        SS_total = np.sum((y - y_bar)**2)
# regression sum of squares
        SS_reg = np.sum((y_hat - y_bar)**2)
# residual sum of squares
        SS_err = np.sum((y - y_hat)**2)
# ====== estimate R2: coefficient of determination ======
        R2 = SS_reg / SS_total
# ====== R2 = correlation_coefficient**2 ======
        correlation_coefficient = np.corrcoef(x, y)
        delta = correlation_coefficient[0, 1]**2 - R2
# ====== estimate MS ======
# sample variance
        MS_total = SS_total / (n-1)
        MS_reg = SS_reg / 1.0
        MS_err = SS_err / (n-2)
# ====== estimate F statistic ======
        F = MS_reg / MS_err
        F_test_p_value = 1 - stats.f._cdf(F, dfn=1, dfd=n-2)
        print("The F value is",F)
# ====== beta_1_hat statistic ======
        beta_1_hat_var = sigma_hat**2 / ((n-1) * np.var(x))
        beta_1_hat_sd = np.sqrt(beta_1_hat_var)
        print("Theta1 variance and stdDev is ", beta_1_hat_var,beta_1_hat_sd)
# confidence interval
        z = stats.t.ppf(q=0.025, df=n-2)# This actually is a t test, since n is small.
        print("z is ", z)
        beta_1_hat_CI_lower_bound = beta_1_hat - z * beta_1_hat_sd
        beta_1_hat_CI_upper_bound = beta_1_hat + z * beta_1_hat_sd
# hypothesis tests for beta_1_hat
# H0: beta_1 = 0
# H1: beta_1 != 0
        beta_1_hat_t_statistic = beta_1_hat / beta_1_hat_sd
        beta_1_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(beta_1_hat_t_statistic), df=n-2))
# ====== beta_0_hat statistic ======
        beta_0_hat_var = beta_1_hat_var * np.sum(x**2) / n # this will be p value
        print("p val is", beta_0_hat_var)
        beta_0_hat_sd = np.sqrt(beta_0_hat_var)
# confidence interval
        beta_0_hat_CI_lower_bound = beta_0_hat - z * beta_0_hat_sd
        beta_1_hat_CI_upper_bound = beta_0_hat + z * beta_0_hat_sd
        beta_0_hat_t_statistic = beta_0_hat / beta_0_hat_sd
        beta_0_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(beta_0_hat_t_statistic), df=n-2))
# confidence interval for the regression line
        sigma_i = 1.0/n * (1 + ((x - x_bar) / np.std(x))**2) # this is also an inportant variable to decribe estimate values, or fn(x) accuracy
        y_hat_sd = sigma_hat * sigma_i
        y_hat_CI_lower_bound = y_hat - z * y_hat_sd
        y_hat_CI_upper_bound = y_hat + z * y_hat_sd
    
        lm_result = {}
        lm_result['beta_1_hat'] = beta_1_hat
        lm_result['beta_0_hat'] = beta_0_hat
        lm_result['sigma_hat'] = sigma_hat
        lm_result['y_hat'] = y_hat
        lm_result['R2'] = R2
        lm_result['F_statistic'] = F
        lm_result['F_test_p_value'] = F_test_p_value
        lm_result['MS_error'] = MS_err
        lm_result['beta_1_hat_CI'] = np.array([beta_1_hat_CI_lower_bound, beta_1_hat_CI_upper_bound])
        lm_result['beta_1_hat_standard_error'] = beta_1_hat_sd
        lm_result['beta_1_hat_t_statistic'] = beta_1_hat_t_statistic
        lm_result['beta_1_hat_t_test_p_value'] = beta_1_hat_t_test_p_value
        lm_result['beta_0_hat_standard_error'] = beta_0_hat_sd
        lm_result['beta_0_hat_t_statistic'] = beta_0_hat_t_statistic
        lm_result['beta_0_hat_t_test_p_value'] = beta_0_hat_t_test_p_value
        lm_result['y_hat_CI_lower_bound'] = y_hat_CI_lower_bound
        lm_result['y_hat_CI_upper_bound'] = y_hat_CI_upper_bound
        return lm_result
# ---------------------------------------------------------------------- end do linear model
# Constructs 
    def __init__(self, *args):
        if (len(args) == 2):    #this should be the normal run time call 
            x = np.array(args[0])
            y = np.array(args[1])   
            newXY = np.array([y[:],x[:]])
            index= np.sort(np.random.randint(0,len(x)-1,size=20))
# Generate random index, sorted, then truncates full dataset into this training set, thryArry
            thryArry = newXY[:,index] 
# Predicting from 20 random datum pairs in original data           
            training_lm_d_result = self.d_lm(thryArry[1],thryArry[0])
            y_theoretical = training_lm_d_result['beta_0_hat'] + training_lm_d_result['beta_1_hat'] * x
# Did not test, but if you enter 4 numbers the auto simulator should run. I guess
        elif(len(args) == 4):
            n = args[0]
            beta_0 = args[1]
            beta_1 = args[2]
            sigma = args[3]
            
            x = -2 + 4 * np.random.rand(n)
            x = np.sort(x)
            epsilon = sigma * np.random.normal(loc=0.0, scale=1, size=n)
            y_theoretical = beta_0 + beta_1 * x
            y = beta_0 + beta_1 * x + epsilon
# One of the original funcitons of this program to simulate data, runs with no input 
        else:
            print("Check2")
            n = 100
            x = -2 + 4 * np.random.rand(n)
            x = np.sort(x)
            beta_0 = 5.0
            beta_1 = 1.5
            sigma = 0.1
        
            epsilon = sigma * np.random.normal(loc=0.0, scale=1, size=n)
            y_theoretical = beta_0 + beta_1 * x
            y = beta_0 + beta_1 * x + epsilon
# --------------------------------------------------------------------------
# linear regression
# --------------------------------------------------------------------------
        n = len(x)
        x_bar = np.mean(x)
        y_bar = np.mean(y) 
# do linear regression using my own function
        lm_d_result = self.d_lm(x[:], y[:])
# plotting section
# --------------------------------------------------------------------------
# set up plotting parameters
# --------------------------------------------------------------------------
        line_width_1 = 2
        line_width_2 = 2
        marker_1 = '.' # point
        marker_2 = 'o' # circle
        marker_size = 12
        line_style_1 = ':' # dotted line
        line_style_2 = '-' # solid line
        fig = plt.figure()
# --------------------------------------------------------------------------
# Position for legend: upper right, upper left, lower right, lower left, center left, center right, upper center, lower center   
# `' Ploter of confidence intervals(forLoops), predicted LR models(plot) and data plots(scatter). 
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x, y, color='red', marker=marker_1, linewidth=line_width_1)
        ax.scatter(thryArry[1], thryArry[0], color='yellow', marker=marker_1, linewidth=line_width_1)
        ax.plot(x, y_theoretical, color='green', label='theoretical', linewidth=line_width_1)
        ax.plot(x, lm_d_result['y_hat'], color='blue', label='predicted', linewidth=line_width_1)
        ax.plot(x, np.ones(n)*y_bar, color='black', linestyle=':', linewidth=line_width_1)
        ax.plot([x_bar, x_bar], [np.min(y), np.max(y)], color='black', linestyle=':', linewidth=line_width_1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title("Linear regression with CI of the regression line")
            
        for i in range(n):
            ax.plot([x[i], x[i]],[lm_d_result['y_hat_CI_lower_bound'][i], 
            lm_d_result['y_hat_CI_upper_bound'][i]], color='magenta',linewidth=line_width_1)   
        for i in range(len(thryArry[0])):
            ax.plot([thryArry[1][i], thryArry[1][i]],[training_lm_d_result['y_hat_CI_lower_bound'][i], 
            training_lm_d_result['y_hat_CI_upper_bound'][i]], color='black',linewidth=line_width_1)
            
        fig.show()
        from time import sleep
        sleep(2) # Time (s) to wait so humans can view lines before sklearn runs
# do linear regression using sklearn, this is general 
        lm_sklearn= linear_model.LinearRegression()
        x_reshaped = x.reshape((len(x), 1))
        lm_sklearn.fit(x_reshaped, y)
        y_hat = lm_sklearn.predict(x_reshaped)
        ax.plot(x, y_hat, color='orange', label='sklearn', linewidth=line_width_1)
        lm_sklearn_result = {}
        lm_sklearn_result['beta_0_hat'] = lm_sklearn.intercept_
        lm_sklearn_result['beta_1_hat'] = lm_sklearn.coef_
        lm_sklearn_result['R2'] = r2_score(y, y_hat)
        lm_sklearn_result['mean_squared_error'] = mean_squared_error(y, y_hat)
        lm_sklearn_result['y_hat'] = y_hat   
        ax.legend(loc='lower right', fontsize=9)  
# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
# 1. are r and y_hat uncorrelated?
# This works, but clutters the current figure. 
        # r = lm_d_result['y_hat'] -  lm_sklearn_result['y_hat']
        # np.corrcoef(r,  lm_sklearn_result['y_hat'])
        # plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(r,lm_sklearn_result['y_hat'], color='blue')
        # ax.set_xlabel('y_hat')
        # ax.set_ylabel('r')
        # fig.show
     
# -----------------------------------------------    run time                           

diabetes = datasets.load_diabetes() 
x=diabetes.data[:,2]
y=diabetes.target
foo2 = do_LinRegAnal(x,y) #Note that this is a constructor call, it's not always reasonable to do this
print("The purple bars on the predicted line are Error, as well as the black bars on the green line")

