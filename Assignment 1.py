#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date:
    Created on Wed Jan 10 09:46:11 2024

Purpose:
    Assignment 1 of Predication & Forecasting
    
Author:
    Thao Le
    Yuanyuan Su
    Jason Wang
"""

###########################################################
### imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###########################################################
### fData()
def fData(lNames):
    
    dfBike = pd.read_excel(lNames[0], index_col = 0)
    dfGas1 = pd.read_excel(lNames[1], index_col = 0)
    dfGas2 = pd.read_excel(lNames[2], index_col = 0)
    dfUmbrella = pd.read_excel(lNames[3])
    dfData = pd.read_excel(lNames[4])
    
    return dfBike, dfGas1, dfGas2, dfUmbrella, dfData

###########################################################
### fEvaluation()
def fEvaluation(vYt, vYt_hat):
    
    vUt = vYt[1: ] - vYt_hat
    dME = round(np.mean(vUt), 2)
    dMAE = round(np.mean(np.abs(vUt)), 2)
    dMAPE = round(100 * np.mean(np.divide(np.abs(vUt), np.abs(vYt[1: ]))), 2)
    dMSE = round(np.mean(vUt ** 2), 2)
    
    return dME, dMAE, dMAPE, dMSE

###########################################################
### fRW()
def fRW(vYt):
    
    vYt_hat = vYt[: -1]
    
    return vYt_hat


###########################################################
### fRW()
def fRA(vYt):
    
    vYt_hat = np.divide(np.cumsum(vYt), np.array(range(1, len(vYt) + 1)))[: -1]
    
    return vYt_hat

###########################################################
### fES()
def fES(vYt, dAlpha):
    
    vYt_hat = np.zeros(len(vYt))
    vYt_hat[0] = vYt[0]
    for i in range(1, len(vYt)):
        vYt_hat[i] = dAlpha * vYt[i - 1] + (1 - dAlpha) * vYt_hat[i - 1]
    
    return vYt_hat[1: ]

###########################################################
### fPlot1()
def fPlot1(dfGas1):
    
    plt.figure(dpi = 300)
    plt.plot(dfGas1, color = 'red')
    plt.show()
    
    return 

###########################################################
### fPlot2()
def fPlot2(dfGas1):
    
    plt.figure(dpi = 300)
    plt.plot(dfGas1, color = 'red')
    plt.axhline(np.mean(dfGas1), color = 'blue')
    plt.show()
    
    return 

###########################################################
### fRA_Plot3()
def fRA_Plot3(dfGas1):
    
    vRA = np.divide(np.cumsum(dfGas1.values), np.array(range(1, len(dfGas1) + 1)))
    plt.figure(dpi = 300)
    plt.plot(dfGas1, color = 'red')
    plt.plot(np.array(range(1, len(dfGas1) + 1)), vRA, color = 'blue')
    plt.show()
    
    return 

###########################################################
### fRA_Plot3()
def fRA_Plot4(dfGas1):
    
    vRA = np.divide(np.cumsum(dfGas1.values), np.array(range(1, len(dfGas1) + 1)))
    
    fig = plt.figure(dpi = 300)
    ax1 = fig.add_subplot(211)
    ax1.plot(dfGas1, color = 'red')
    ax1.plot(np.array(range(2, len(dfGas1) + 1)), vRA[: -1], color = 'blue')
    
    ax2 = fig.add_subplot(212)
    ax2.vlines(np.array(range(1, len(dfGas1) + 1)), 0, np.insert(dfGas1['Gasoline'].values[1: ] - vRA[: -1], 0, 0))
    ax2.axhline(0)
    ax2.scatter(np.array(range(2, len(dfGas1) + 1)), dfGas1['Gasoline'].values[1: ] - vRA[: -1])
    plt.tight_layout(pad = 1.08)
    plt.show()
    
    return vRA

###########################################################
### fRA_Plot5()
def fPlot5_6_7_8(dfGas1):
    
    vRAweights = np.array([1 / 50] * 50)
    vMAweights = np.zeros(50)
    vRWweights = np.zeros(50)
    vESweights = np.zeros(50)
    vMAweights[: 10] = 1 / 10
    vRWweights[0] = 1
    dAlpha = 0.2
    vESweights[-1] = (1 - dAlpha) ** 49
    vESweights[: -1] = dAlpha * np.power((1 - dAlpha), np.array(range(50 - 1)))
    
    fig = plt.figure(dpi = 300)
    ax1 = fig.add_subplot(221)
    ax1.bar(np.array(range(50)), vRAweights, label = 'Running Average')
    ax1.set_ylim(0, 0.03)
    ax1.legend()
    ax2 = fig.add_subplot(222)
    ax2.bar(np.array(range(50)), vMAweights, label = 'Moving Average')
    ax2.legend()
    ax3 = fig.add_subplot(223)
    ax3.bar(np.array(range(50)), vRWweights, label = 'Random Walk')
    ax3.legend()
    ax4 = fig.add_subplot(224)
    ax4.bar(np.array(range(50)), vESweights, label = 'Exp Smo')
    ax4.legend()
    plt.tight_layout(pad = 1.08)
    plt.show()
    
    fig = plt.figure(dpi = 300)
    ax1 = fig.add_subplot(221)
    dAlpha = 0.05
    vESweights[-1] = (1 - dAlpha) ** 49
    vESweights[: -1] = dAlpha * np.power((1 - dAlpha), np.array(range(50 - 1)))
    ax1.bar(np.array(range(50)), vESweights, label = 'alpha = 0.05')
    ax1.legend()
    dAlpha = 0.1
    vESweights[-1] = (1 - dAlpha) ** 49
    vESweights[: -1] = dAlpha * np.power((1 - dAlpha), np.array(range(50 - 1)))
    ax2 = fig.add_subplot(222)
    ax2.bar(np.array(range(50)), vESweights, label = 'alpha = 0.10')
    ax2.legend()
    dAlpha = 0.2
    vESweights[-1] = (1 - dAlpha) ** 49
    vESweights[: -1] = dAlpha * np.power((1 - dAlpha), np.array(range(50 - 1)))
    ax3 = fig.add_subplot(223)
    ax3.bar(np.array(range(50)), vESweights, label = 'alpha = 0.20')
    ax3.legend()
    dAlpha = 0.5
    vESweights[-1] = (1 - dAlpha) ** 49
    vESweights[: -1] = dAlpha * np.power((1 - dAlpha), np.array(range(50 - 1)))
    ax4 = fig.add_subplot(224)
    ax4.bar(np.array(range(50)), vESweights, label = 'alpha = 0.50')
    ax4.legend()
    plt.tight_layout(pad = 1.08)
    plt.show()
    
    fig = plt.figure(dpi = 300)
    ax1 = fig.add_subplot(221)
    dAlpha = 0.05
    vESweights = np.power((1 - dAlpha), np.array(range(50)))
    dMI = round(np.log(0.1) / np.log(1 - dAlpha), 2)
    ax1.bar(np.array(range(50)), vESweights, label = 'alpha = 0.05')
    ax1.axhline(0.1, color = 'red', label = '0.1, mem idx = {}'.format(dMI))
    ax1.legend()
    dAlpha = 0.1
    vESweights = np.power((1 - dAlpha), np.array(range(50)))
    dMI = round(np.log(0.1) / np.log(1 - dAlpha), 2)
    ax2 = fig.add_subplot(222)
    ax2.bar(np.array(range(50)), vESweights, label = 'alpha = 0.10')
    ax2.axhline(0.1, color = 'red', label = '0.1, mem idx = {}'.format(dMI))
    ax2.legend()
    dAlpha = 0.2
    vESweights = np.power((1 - dAlpha), np.array(range(50)))
    dMI = round(np.log(0.1) / np.log(1 - dAlpha), 2)
    ax3 = fig.add_subplot(223)
    ax3.bar(np.array(range(50)), vESweights, label = 'alpha = 0.20')
    ax3.axhline(0.1, color = 'red', label = '0.1, mem idx = {}'.format(dMI))
    ax3.legend()
    dAlpha = 0.5
    vESweights = np.power((1 - dAlpha), np.array(range(50)))
    dMI = round(np.log(0.1) / np.log(1 - dAlpha), 2)
    ax4 = fig.add_subplot(224)
    ax4.bar(np.array(range(50)), vESweights, label = 'alpha = 0.50')
    ax4.axhline(0.1, color = 'red', label = '0.1, mem idx = {}'.format(dMI))
    ax4.legend()
    plt.tight_layout(pad = 1.08)
    plt.show()
    
    return 

###########################################################
### fPlot9()
def fPlot9_10(dfGas1):
    
    vYt = dfGas1['Gasoline'].values
    lAlpha = [0.2, 0.8]
    for dAlpha in lAlpha:
        vES = fES(vYt, dAlpha)
        
        fig = plt.figure(dpi = 300)
        ax1 = fig.add_subplot(211)
        ax1.plot(dfGas1, color = 'red')
        ax1.plot(np.array(range(2, len(dfGas1) + 1)), vES, color = 'blue')
        
        ax2 = fig.add_subplot(212)
        ax2.vlines(np.array(range(1, len(dfGas1) + 1)), 0, np.insert(dfGas1['Gasoline'].values[1: ] - vES, 0, 0))
        ax2.axhline(0)
        ax2.scatter(np.array(range(2, len(dfGas1) + 1)), dfGas1['Gasoline'].values[1: ] - vES)
        plt.tight_layout(pad = 1.08)
        plt.show()
    
    return 
###########################################################
### fTable1_2()
def fTable1_2(dfGas1):
    
    vRA = np.divide(np.cumsum(dfGas1.values), np.array(range(1, len(dfGas1) + 1)))
    vYt = dfGas1['Gasoline'].values[1: ]
    vYt_hat = vRA[: -1]
    vUt = vYt - vYt_hat
    dfTable = pd.DataFrame(np.stack([vYt, vYt_hat, vUt]).T, columns = ['Y_t', 'Y_t_hat', 'U_t'])
    dfTable1 = pd.concat([dfTable, pd.DataFrame(np.mean(dfTable, axis = 0)).T], axis = 0)
    
    dfTable2 = dfTable.copy()
    dfTable2['|U_t|'] = np.abs(dfTable['U_t'])
    dfTable2['%|U_t|'] = 100 * dfTable2['|U_t|'] / dfTable2['Y_t']
    dfTable2['U_t2'] = dfTable2['U_t'] ** 2
    dfTable2 = np.round(dfTable2, 2)
    dME, dMAE, dMAPE, dMSE = fEvaluation(vYt, vUt)
    
    return dfTable1, dfTable2

###########################################################
### fTable3()
def fTable3_4(dfGas1):
    
    vYt = dfGas1['Gasoline'].values
    vRA = fRA(vYt)
    vRW = fRW(vYt)
    dfTable3 = pd.DataFrame(np.vstack([fEvaluation(vYt, vRA), fEvaluation(vYt, vRW)]), columns = ['ME', 'MAE', 'MAPE', 'MSE'], index = ['Running Avg', 'Random Walk'])
    dfTable4 = pd.DataFrame(np.vstack([fEvaluation(vYt[5: ], vRA[5: ]), fEvaluation(vYt[5: ], vRW[5: ])]), columns = ['ME', 'MAE', 'MAPE', 'MSE'], index = ['Running Avg', 'Random Walk'])
    
    return dfTable3, dfTable4

###########################################################
### fTable1()
def fTable5(dfGas1):
    
    vYt = dfGas1['Gasoline'].values
    vRA = fRA(vYt)
    vRW = fRW(vYt)
    vES1 = fES(vYt, dAlpha = 0.2)
    vES2 = fES(vYt, dAlpha = 0.8)
    dfTable5 = pd.DataFrame(np.vstack([fEvaluation(vYt[5: ], vRA[5: ]), fEvaluation(vYt[5: ], vRW[5: ]), fEvaluation(vYt[5: ], vES1[5: ]), fEvaluation(vYt[5: ], vES2[5: ])]), columns = ['ME', 'MAE', 'MAPE', 'MSE'], index = ['Running Avg', 'Random Walk', 'ExpSmo (0.2)', 'ExpSmo (0.8)'])
    
    return dfTable5

###########################################################
### fData()
def fPredict(dfGas1):
    
    plt.figure(dpi = 300)
    plt.plot(dfGas1, color = 'red')
    plt.axhline(np.mean(dfGas1), color = 'blue')
    plt.show()

    return



###########################################################
### main()
def main():
    
    # Import datasets
    lNames = ['BicycleSales.xlsx', 'GasolineSales1.xlsx', 'GasolineSales2.xlsx', 'Umbrella.xlsx', 'DataAssignment1.xlsx']
    dfBike, dfGas1, dfGas2, dfUmbrella, dfData = fData(lNames)
    
    # Question (a)
    
    
    
    
    
    
    
    
    
    
    
###########################################################
### start main
if __name__ == "__main__":
    main()

