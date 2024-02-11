#-------------------------------------------------------------------------------
# Name:        Distance and Inputs Graphs with respect to R2 score
# Purpose:     Creating Distance, P10,temp,humidity,press,windpeed and direction
#with respect to R2_score for evaluation
# Author:      Caleb JUma
#
# Created:     15-02-2020
# Copyright:   (c) t-82maca1mpg 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# Imports


# Data Manipulation
import numpy as np
import pandas as pd
import random
import math
from collections import Counter

# Files/OS
import os
import copy

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import logistic

# Benchmarking
import time

# Error Analysis
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

matplotlib.rcParams.update({'font.size': 14})

##df = pd.read_csv('C:/Users/juma/Desktop/for abstract/abstracctt/sensorIDs_output.csv')#put in a diff folder to the py file
##print(df.head())

dfb = pd.read_csv('C:/Users/t-82maca1mpg/Desktop/MSCPG_CalebJuma/RNN/metrics/Sensor_81_IDs_Metrics_adjusted.csv')#put in a diff folder to the py file
print(dfb.head())
##dfb = pd.read_csv('C:/Users/t-82maca1mpg/Desktop/MSCPG_CalebJuma/RNN/metrics/Sensor_81_IDs_Metrics.csv')#put in a diff folder to the py file
##print(dfb.head())

##df_train = df.iloc[:,0]
##df_train2=df.iloc[:,1]

df_train = dfb.iloc[:(-1),2]#R2 score (-1 is to prevent us using the base sensor in the analysis. the base sensor is the last row.)

##x axis
##df_train2=dfb.iloc[:(-1),5]#distance
##df_train2=dfb.iloc[:(-1),6]#p2.5 corr coeff
##df_train2=dfb.iloc[:(-1),7]#p10 corr coeff
##df_train2=dfb.iloc[:(-1),8]#temp corr coeff
##df_train2=dfb.iloc[:(-1),9]#humidity corr coeff
##df_train2=dfb.iloc[:(-1),10]#pressure corr coeff
##df_train2=dfb.iloc[:(-1),11]#wind direction corr coeff
df_train2=dfb.iloc[:(-1),12]# windspeed corr coeff

listt=[]
listt2=[]


for i in df_train:
    listt.append(i)

for i in df_train2:
    listt2.append(i)


#(μg)

ydata = np.asarray(listt, dtype=np.float32)
xdata = np.asarray(listt2, dtype=np.float32)

def func(x, a, b, c):
    return a * np.log(b * x) + c


def funcline(x, a, b):
    return a*x + b

popt, pcov = curve_fit(func, xdata, ydata)
poptline, pcovline = curve_fit(funcline, xdata, ydata)

##fig=plt.figure()

##plt.scatter(listt2,listt,label="Sensors")
##plt.plot(xdata, funcline(xdata, *poptline), 'r-', label="Fitted Curve")
##plt.title("R² Score per Distance")
##plt.ylabel("R² score")
##plt.xlabel("Distance (m) from Base sensor")
##plt.legend()
##plt.show()


##fig=plt.figure()
##plt.scatter(listt2,listt,label="Sensors")
##plt.plot(xdata, func(xdata, *popt), 'r-', label="Fitted Curve")
##plt.title("R² Score per P2.5 ρ")
##plt.ylabel("R² score")
##plt.xlabel("P2.5 coefficient of correlation(ρ)  with respect to Base sensor")
##plt.legend()
##plt.show()

##fig=plt.figure()
##plt.scatter(listt2,listt,label="Sensors")
##plt.plot(xdata, funcline(xdata, *poptline), 'r-', label="Fitted Curve")
##plt.title("R² Score per P10 ρ")
##plt.ylabel("R² score")
##plt.xlabel("P10 coefficient of correlation(ρ)  with respect to Base sensor")
##plt.legend()
##plt.show()



##fig=plt.figure()
##plt.scatter(listt2,listt,label="Sensors")
##plt.plot(xdata, funcline(xdata, *poptline), 'r-', label="Fitted Curve")
##plt.title("R² Score per Temperature ρ")
##plt.ylabel("R² score")
##plt.xlabel("Temperature coefficient of correlation(ρ)  with respect to Base sensor")
##plt.legend()
##plt.show()
##
####

##fig=plt.figure()
##plt.scatter(listt2,listt,label="Sensors")
##plt.plot(xdata, funcline(xdata, *poptline), 'r-', label="Fitted Curve")
##plt.title("R² Score per Humidity ρ")
##plt.ylabel("R² score")
##plt.xlabel("Humidty coefficient of correlation(ρ)  with respect to Base sensor")
##plt.legend()
##plt.show()


##
##fig=plt.figure()
##plt.scatter(listt2,listt,label="Sensors")
##plt.plot(xdata, funcline(xdata, *poptline), 'r-', label="Fitted Curve")
##plt.title("R² Score per Pressure ρ")
##plt.ylabel("R² score")
##plt.xlabel("Pressure coefficient of correlation(ρ)  with respect to Base sensor")
##plt.legend()
##plt.show()


##
##fig=plt.figure()
##plt.scatter(listt2,listt,label="Sensors")
##plt.plot(xdata, funcline(xdata, *poptline), 'r-', label="Fitted Curve")
##plt.title("R² Score per wind direction ρ")
##plt.ylabel("R² score")
##plt.xlabel("wind direction coefficient of correlation(ρ)  with respect to Base sensor")
##plt.legend()
##plt.show()

fig=plt.figure()
plt.scatter(listt2,listt,label="Sensors")
plt.plot(xdata, funcline(xdata, *poptline), 'r-', label="Fitted Curve")
plt.title("R² Score per wind speed ρ")
plt.ylabel("R² score")
plt.xlabel("Windspeed coefficient of correlation  with respect to Base sensor")
plt.legend()
plt.show()


