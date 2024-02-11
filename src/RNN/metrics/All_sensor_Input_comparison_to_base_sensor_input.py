#-------------------------------------------------------------------------------
# Name:        P10 comparisons
# Purpose:     Comparing P10 of all sensors to base sensor P10
#
# Author:      Caleb JUma
#
# Created:     15-02-2020
# Copyright:   (c) t-82maca1mpg 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# Imports
from statistics import mean

# Data Manipulation
import numpy as np
import pandas as pd
import random
import math

# Files/OS
import os
import copy

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Benchmarking
import time

# Error Analysis
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#outputting results on excel table
from openpyxl import Workbook


matplotlib.rcParams.update({'font.size': 15})
##################################################################################################################
##THe if statement and the vscor and tscor are used a way of measuring the accuracy of each...
#...RNN variant with respect to the final Variance score and time taken.
#They can be removed without affecting the process.

#list with timestamp, sensorID, P10 and correlation coefficient
lab1=[]#sensorId
lab2=[]#P2_correlation coefficient
lab3=[]#P10 correlation coefficient
lab4=[]#temp correlation coefficient
lab5=[]#humidity correlation coefficient
lab6=[]#pressure correlation coefficient
lab7=[]#windspeed correlation coefficient
lab8=[]#winddire correlation coefficient



#Input Sensors
SensorIDa=[140]

##SensorIDa=[140,189,209,215,271,286,287,295,438,466,535,547,585,665,671,673,757,773,\
##789,793,795,986,1148,1186,1282,1356,1434,1483,2199,2299,2480,2586,2590,2870,3559,4636,\
##4827,4837,5127,5756,5929,6479,6582,7003,7037,7100,7150,7193,7497,7561,7895,8183,8349,8458,\
##8881,8938,9485,9900,10311,10418,10529,10548,10573,11285,13747,13764,13766,14205,\
##14580,15329,16195,18332,18425,18702,19863,21943,22280,22926,23302,23780,25341]

df = pd.read_csv('C:/Users/t-82maca1mpg/Desktop/MSCPG_CalebJuma/RNN/data in csv/sensorID_5127_csv_table_sorted.csv')#put in a diff folder to the py file
print(df.head())
df_train = df.iloc[:,1]
for i in SensorIDa:
    dfb = pd.read_csv('C:/Users/t-82maca1mpg/Desktop/MSCPG_CalebJuma/RNN/data in csv/sensorID_{}_csv_table_sorted.csv'.format(i))#put in a diff folder to the py file
    print(dfb.head())
    corrcoff_all=df.corrwith(dfb,axis=0)
    print(corrcoff_all)
    corrcoff=pd.DataFrame.from_dict(corrcoff_all)
##    print(corrcoff)
    df_p10 = corrcoff.iat[1,0]
    df_p2= corrcoff.iat[2,0]
    df_temp = corrcoff.iat[4,0]
    df_hum = corrcoff.iat[5,0]
    df_pres = corrcoff.iat[6,0]
    df_winddsp = corrcoff.iat[7,0]
    df_winddire = corrcoff.iat[8,0]

    #create temprary lists
    lab1.append(i)
    lab2.append(df_p2)
    lab3.append(df_p10)
    lab4.append(df_temp)
    lab5.append(df_hum)
    lab6.append(df_pres)
    lab7.append(df_winddsp)
    lab8.append(df_winddire)



##ebook_p10comp=pd.DataFrame({'Sensor_ID':lab1,'P2_5_corr':lab2,'P10_corr':lab3,'temperature_corr':lab4,'humidity_corr':lab5,'Pressure_corr':lab6,'windspeed_corr':lab7,'winddirection_corr':lab8,})
##ebook_p10comp.to_excel('C:/Users/t-82maca1mpg/Desktop/MSCPG_CalebJuma/RNN/metrics/SensorIDs_Input_comparison_result.xlsx',sheet_name='sheet1',index=False)
