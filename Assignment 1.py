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
from statsmodels.tsa.ar_model import AutoReg

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
### fRT()
def fRT(vYt):
    
    vYt_hat = np.zeros(len(vYt) - 2)
    for i in range(len(vYt_hat)):
        mX = np.vstack([np.ones(i + 2), np.array(range(1, i + 3))]).T
        vY = vYt[: i + 2]
        vBeta = np.linalg.inv(mX.T @ mX) @ mX.T @ vY
        vYt_hat[i] = np.array([1, i + 3]) @ vBeta
    
    return vYt_hat

###########################################################
### fRW_Drift()
def fRW_Drift(vYt):
    
    vYt_hat = np.zeros(len(vYt) - 2)
    vDiff = np.diff(vYt)
    for i in range(len(vYt_hat)):
        dC = np.mean(vDiff[: i + 1])
        print(dC)
        vYt_hat[i] = vYt[i + 1] + dC
    
    return vYt_hat

###########################################################
### fHolt_Winters()
def fHolt_Winters(vYt, dAlpha, dBeta):
    
    vYt_hat = np.zeros(len(vYt) - 2)
    dL = vYt[0]
    dG = vYt[1] - vYt[0]
    for i in range(len(vYt_hat)):
        dL_new = dAlpha * vYt[i + 1] + (1 - dAlpha) * (dL + dG)
        dG = dBeta * (dL_new - dL) + (1 - dBeta) * dG
        vYt_hat[i] = dL_new + dG
        dL = dL_new.copy()
    
    return vYt_hat

###########################################################
### fSeasonal_RW_Drift()
def fSeasonal_RW_Drift(vYt, sSeason):
    
    iN = len(vYt)
    if sSeason == 'seasonal':
        iS = 4
    elif sSeason == 'monthly':
        iS = 12
    vYt_hat = np.zeros(iN - iS - 1)
    for i in range(len(vYt_hat)):
        dSum = 0
        for j in range(i + 1):
            dSum = dSum + vYt[i + iS + 1 - j] - vYt[i + 1 - j]
        dC = 1 / (i + iS + 1) * dSum
        vYt_hat[i] = vYt[i + 1] + dC
        
    return vYt_hat

###########################################################
### fRunning_SR()
def fRunning_SR(vYt, vSeason, sSeason):
    
    iN = len(vYt)
    if sSeason == 'seasonal':
        iS = 4
    elif sSeason == 'monthly':
        iS = 12
    vYt_hat = np.zeros(iN - iS - 1)
    mSeason = np.vstack((np.diag(np.ones(iS - 1)), -np.ones(iS - 1)))    
    mX_part1 = np.vstack([np.ones(iN), np.array(range(1, iN + 1))]).T
    mX_part2 = np.vstack((np.tile(mSeason, (iN // iS, 1)), mSeason[: iN % iS]))
    mX = np.hstack((mX_part1, mX_part2))
    
    for i in range(len(vYt_hat)):
        mXt = mX[: i + iS + 1]
        vY = vYt[: i + iS + 1]
        vBeta = np.linalg.inv(mXt.T @ mXt) @ mXt.T @ vY
        vYt_hat[i] = mX[i + iS + 1] @ vBeta
        
    return vYt_hat

###########################################################
### fSeasonal_HW_Multi()
def fSeasonal_HW_Multi(vYt, sSeason, dAlpha, dBeta, dGamma):
    
    iN = len(vYt)
    if sSeason == 'seasonal':
        iS = 4
    elif sSeason == 'monthly':
        iS = 12
    vYt_hat = np.zeros(iN - iS - 1)
    dLt = np.mean(vYt[: iS])
    dGt = (np.mean(vYt[iS: 2 * iS]) - dLt) / iS
    vHt = np.zeros(iN)
    vHt[: iS] = vYt[: iS] / dLt
    
    for i in range(len(vYt_hat)):
        dLt_new = dAlpha * (vYt[i + iS] / vHt[i]) + (1 - dAlpha) * (dLt + dGt)
        dGt = dBeta * (dLt_new - dLt) + (1 - dBeta) * dGt
        vHt[iS + i] = dGamma * vYt[i + iS] / dLt_new + (1 - dGamma) * vHt[i]
        vYt_hat[i] = vHt[i + 1] * (dLt_new + dGt)
        dLt = dLt_new
    vYt_hat = np.round(vYt_hat, 2)
        
    return vYt_hat

###########################################################
### fSeasonal_HW_Multi()
def fSeasonal_HW_Add(vYt, sSeason, dAlpha, dBeta, dGamma):
    
    iN = len(vYt)
    if sSeason == 'seasonal':
        iS = 4
    elif sSeason == 'monthly':
        iS = 12
    vYt_hat = np.zeros(iN - iS - 1)
    dLt = np.mean(vYt[: iS])
    dGt = (np.mean(vYt[iS: 2 * iS]) - dLt) / iS
    vHt = np.zeros(iN)
    vHt[: iS] = vYt[: iS] - dLt
    
    for i in range(len(vYt_hat)):
        dLt_new = dAlpha * (vYt[i + iS] / vHt[i]) + (1 - dAlpha) * (dLt + dGt)
        dGt = dBeta * (dLt_new - dLt) + (1 - dBeta) * dGt
        vHt[iS + i] = dGamma * vYt[i + iS] / dLt_new + (1 - dGamma) * vHt[i]
        vYt_hat[i] = vHt[i + 1] + dLt_new + dGt
        dLt = dLt_new
    vYt_hat = np.round(vYt_hat, 2)
        
    return vYt_hat

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
    plt.axhline(np.mean(np.mean(dfGas1['Gasoline'])), color = 'blue')
    plt.show()
    
    return 

###########################################################
### fPlot3()
def fPlot3(dfGas1):
    
    vRA = np.divide(np.cumsum(dfGas1.values), np.array(range(1, len(dfGas1) + 1)))
    plt.figure(dpi = 300)
    plt.plot(dfGas1, color = 'red')
    plt.plot(np.array(range(1, len(dfGas1) + 1)), vRA, color = 'blue')
    plt.show()
    
    return 

###########################################################
### fPlot4()
def fPlot4(dfGas1):
    
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
### fPlot9_10()
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
### fPlot12()
def fPlot12(dfGas2):
    
    plt.figure(dpi = 300)
    plt.plot(dfGas2, color = 'red')
    plt.show()
    
    return 

###########################################################
### fPlot12()
def fPlot13(dfGas2):
    
    vYt = dfGas2['Gasoline'].values
    vRA = fRA(vYt)
    vRW = fRW(vYt)
    vES = fES(vYt, dAlpha = 0.2)
    vAR = np.zeros(len(vYt) - 7)
    for i in range(len(vYt) - 7):
        AR_model = AutoReg(vYt[: 7], lags = 1).fit()
        vAR = AR_model.predict(7, len(vYt))
    
    fig = plt.figure(dpi = 300)
    ax1 = fig.add_subplot(321)
    ax1.plot(np.array(range(1, len(vYt) + 1)), vYt, label = 'Gasoline Sales')
    ax1.axhline(np.mean(vYt[: 12]), color = 'red', label = 'Expert Forecast')
    ax1.legend(prop={'size': 6})
    ax2 = fig.add_subplot(322)
    ax2.plot(np.array(range(1, len(vYt) + 1)), vYt, label = 'Gasoline Sales')
    ax2.plot(np.array(range(2, len(vYt) + 1)), vRA, label = 'Running Avg Forecast')
    ax2.legend(prop={'size': 6})
    ax3 = fig.add_subplot(323)
    ax3.plot(np.array(range(1, len(vYt) + 1)), vYt, label = 'Gasoline Sales')
    ax3.plot(np.array(range(2, len(vYt) + 1)), vRW, label = 'Random Walk Forecast')
    ax3.legend(prop={'size': 6})
    ax4 = fig.add_subplot(324)
    ax4.plot(np.array(range(1, len(vYt) + 1)), vYt, label = 'Gasoline Sales')
    ax4.plot(np.array(range(7, len(vYt) + 1)), vAR, label = 'AR(1) Forecast')
    ax4.legend(prop={'size': 6})
    ax5 = fig.add_subplot(325)
    ax5.plot(np.array(range(1, len(vYt) + 1)), vYt, label = 'Gasoline Sales')
    ax5.plot(np.array(range(2, len(vYt) + 1)), vES, label = 'Exp Smo Forecast, alpha = 0.2')
    ax5.legend(prop={'size': 6})
    plt.tight_layout(pad = 1.08)
    plt.show()
    
    return 

###########################################################
### fTable1_2()
def fTable1_2(dfGas1):
    
    vRA = np.divide(np.cumsum(dfGas1.values), np.array(range(1, len(dfGas1) + 1)))
    vYt = dfGas1['Gasoline']
    vYt_hat = vRA[: -1]
    vUt = vYt[1: ] - vYt_hat
    dfTable = pd.DataFrame(np.stack([vYt[1: ], vYt_hat, vUt]).T, columns = ['Y_t', 'Y_t_hat', 'U_t'])
    dfTable1 = pd.concat([dfTable, pd.DataFrame(np.mean(dfTable, axis = 0)).T], axis = 0)
    
    dfTable2 = dfTable.copy()
    dfTable2['|U_t|'] = np.abs(dfTable['U_t'])
    dfTable2['%|U_t|'] = 100 * dfTable2['|U_t|'] / dfTable2['Y_t']
    dfTable2['U_t2'] = dfTable2['U_t'] ** 2
    dfTable2 = np.round(dfTable2, 2)
    dME, dMAE, dMAPE, dMSE = fEvaluation(vYt, vYt_hat)
    
    return dfTable1, dfTable2

###########################################################
### fTable3_4()
def fTable3_4(dfGas1):
    
    vYt = dfGas1['Gasoline'].values
    vRA = fRA(vYt)
    vRW = fRW(vYt)
    dfTable3 = pd.DataFrame(np.vstack([fEvaluation(vYt, vRA), fEvaluation(vYt, vRW)]), columns = ['ME', 'MAE', 'MAPE', 'MSE'], index = ['Running Avg', 'Random Walk'])
    dfTable4 = pd.DataFrame(np.vstack([fEvaluation(vYt[5: ], vRA[5: ]), fEvaluation(vYt[5: ], vRW[5: ])]), columns = ['ME', 'MAE', 'MAPE', 'MSE'], index = ['Running Avg', 'Random Walk'])
    
    return dfTable3, dfTable4

###########################################################
### fTable5()
def fTable5(dfGas1):
    
    vYt = dfGas1['Gasoline'].values
    vRA = fRA(vYt)
    vRW = fRW(vYt)
    vES1 = fES(vYt, dAlpha = 0.2)
    vES2 = fES(vYt, dAlpha = 0.8)
    dfTable5 = pd.DataFrame(np.vstack([fEvaluation(vYt[5: ], vRA[5: ]), fEvaluation(vYt[5: ], vRW[5: ]), fEvaluation(vYt[5: ], vES1[5: ]), fEvaluation(vYt[5: ], vES2[5: ])]), columns = ['ME', 'MAE', 'MAPE', 'MSE'], index = ['Running Avg', 'Random Walk', 'ExpSmo (0.2)', 'ExpSmo (0.8)'])
    
    return dfTable5

###########################################################
### fTable7()
def fTable7(dfGas2):
    
    vYt = dfGas2['Gasoline'].values
    vRA = fRA(vYt)
    vRW = fRW(vYt)
    vES1 = fES(vYt, dAlpha = 0.2)
    vES2 = fES(vYt, dAlpha = 0.8)
    dfTable7 = pd.DataFrame(np.vstack([fEvaluation(vYt[5: ], vRA[5: ]), fEvaluation(vYt[5: ], vRW[5: ]), fEvaluation(vYt[5: ], vES1[5: ]), fEvaluation(vYt[5: ], vES2[5: ])]), columns = ['ME', 'MAE', 'MAPE', 'MSE'], index = ['Running Avg', 'Random Walk', 'ExpSmo (0.2)', 'ExpSmo (0.8)'])
    
    return dfTable7

###########################################################
### fGasPredict()
def fGasPredict(dfGas1, dfGas2):
    
    fPlot1(dfGas1)
    fPlot2(dfGas1)
    fPlot3(dfGas1)
    fPlot4(dfGas1)
    fPlot5_6_7_8(dfGas1)
    fPlot9_10(dfGas1)
    fPlot12(dfGas2)
    fPlot13(dfGas2)
    dfTable1, dfTable2 = fTable1_2(dfGas1)
    dfTable3, dfTable4 = fTable3_4(dfGas1)
    dfTable5 = fTable5(dfGas1)
    dfTable7 = fTable7(dfGas2)
    
    return dfTable1, dfTable2, dfTable3, dfTable4, dfTable5, dfTable7

###########################################################
### main()
def main():
    
    # Import datasets
    lNames = ['BicycleSales.xlsx', 'GasolineSales1.xlsx', 'GasolineSales2.xlsx', 'Umbrella.xlsx', 'DataAssignment1.xlsx']
    dfBike, dfGas1, dfGas2, dfUmbrella, dfData = fData(lNames)
    
    # Question (a)
    dfTable1, dfTable2, dfTable3, dfTable4, dfTable5, dfTable7 = fGasPredict(dfGas1, dfGas2)
    
    
    
    
    
    
    
    
    
    
###########################################################
### start main
if __name__ == "__main__":
    main()

