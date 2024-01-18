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
    dfSun = pd.read_csv('Sunspot.csv', sep = ';', header = None).iloc[-84:, :][[3]]
    dfSun.columns = ['Sunspot']
    dfSun.index = np.array(range(len(dfSun)))
    
    return dfBike, dfGas1, dfGas2, dfUmbrella, dfData, dfSun

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
    
    return np.round(vYt_hat[1: ], 2)

###########################################################
### fRT()
def fRT(vYt):
    
    vYt_hat = np.zeros(len(vYt) - 2)
    mBeta = np.zeros((len(vYt) - 2, 2))
    for i in range(len(vYt_hat)):
        mX = np.vstack([np.ones(i + 2), np.array(range(1, i + 3))]).T
        vY = vYt[: i + 2]
        vBeta = np.linalg.inv(mX.T @ mX) @ mX.T @ vY
        mBeta[i, :] = vBeta
        vYt_hat[i] = np.array([1, i + 3]) @ vBeta
    
    return np.round(vYt_hat, 2), mBeta

###########################################################
### fRW_Drift()
def fRW_Drift(vYt):
    
    vYt_hat = np.zeros(len(vYt) - 2)
    vCt = np.zeros(len(vYt) - 2)
    vDiff = np.diff(vYt)
    for i in range(len(vYt_hat)):
        dC = np.mean(vDiff[: i + 1])
        vYt_hat[i] = vYt[i + 1] + dC
        vCt[i] = dC
    
    return np.round(vYt_hat, 2), vCt

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
    
    return np.round(vYt_hat, 2)

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
def fRunning_SR(vYt, sSeason):
    
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
### fSeasonal_HW_Add()
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
    ax1.bar(np.array(range(49)), vESweights[: -1], label = 'alpha = 0.05')
    ax1.legend()
    dAlpha = 0.1
    vESweights[-1] = (1 - dAlpha) ** 49
    vESweights[: -1] = dAlpha * np.power((1 - dAlpha), np.array(range(50 - 1)))
    ax2 = fig.add_subplot(222)
    ax2.bar(np.array(range(49)), vESweights[: -1], label = 'alpha = 0.10')
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
    
    # Initialize arrays to store the forecasted values
    start_observation = 6
    vAR1 = np.zeros_like(vYt, dtype=float)
    vAR2 = np.zeros_like(vYt, dtype=float)
    
    # Estmated AR(1) model parameters before adding new observations
    vYt_before = vYt[:12]
    model = AutoReg(vYt_before, lags =1).fit()
    constant = model.params[0]
    coefficient = model.params[1]
    # Use the AR(1) formula to construct the one-step-ahead forecast
    for t in range(start_observation, len(vYt)):
        vAR1[t] = constant + coefficient * vYt[t-1]
    vAR1 = vAR1[start_observation:]

    # Estmate AR(1) model parameters after adding new obs
    model = AutoReg(vYt, lags =1).fit()
    constant = model.params[0]
    coefficient = model.params[1]
    # Use the AR(1) formula to construct the one-step-ahead forecast
    for t in range(start_observation, len(vYt)):
        vAR2[t] = constant + coefficient * vYt[t-1]
    vAR2 = vAR2[start_observation:]
    
    # Merge two forecasts
    vAR = np.zeros_like(vYt[6:], dtype=float)
    vAR[:7] = vAR1[:7]
    vAR[7:] = vAR2[7:]
    
    fig = plt.figure(dpi = 300)
    ax1 = fig.add_subplot(321)
    ax1.plot(np.array(range(1, len(vYt) + 1)), vYt, color = 'red', label = 'Gasoline Sales')
    ax1.axhline(np.mean(vYt[: 12]), color = 'blue', label = 'Expert Forecast')
    ax1.legend(prop={'size': 6})
    ax2 = fig.add_subplot(322)
    ax2.plot(np.array(range(1, len(vYt) + 1)), vYt, color = 'red', label = 'Gasoline Sales')
    ax2.plot(np.array(range(2, len(vYt) + 1)), vRA, color = 'blue', label = 'Running Avg Forecast')
    ax2.legend(prop={'size': 6})
    ax3 = fig.add_subplot(323)
    ax3.plot(np.array(range(1, len(vYt) + 1)), vYt, color = 'red', label = 'Gasoline Sales')
    ax3.plot(np.array(range(2, len(vYt) + 1)), vRW, color = 'blue', label = 'Random Walk Forecast')
    ax3.legend(prop={'size': 6})
    ax4 = fig.add_subplot(324)
    ax4.plot(np.array(range(1, len(vYt) + 1)), vYt, color = 'red', label = 'Gasoline Sales')
    ax4.plot(np.array(range(7, len(vYt) + 1)), vAR, color = 'blue', label = 'AR(1) Forecast')
    ax4.legend(prop={'size': 6})
    ax5 = fig.add_subplot(325)
    ax5.plot(np.array(range(1, len(vYt) + 1)), vYt, color = 'red', label = 'Gasoline Sales')
    ax5.plot(np.array(range(2, len(vYt) + 1)), vES, color = 'blue', label = 'Exp Smo Forecast, alpha = 0.2')
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


################## Bicycle prediction plots
###########################################################
### fPlot14()
def fPlot14(dfBike):
    plt.figure(dpi = 300)
    plt.plot(dfBike["Bicycle"], color = 'red')
    plt.legend(labels=["Bicycle Sales"])
    plt.show()
    return 

###########################################################
### fPlot15()
def fPlot15(dfBike):
    # vRA = np.divide(np.cumsum(dfBike["Bicycle"].values), np.array(range(1, len(dfBike) + 1)))
    vYt = dfBike["Bicycle"].values
    vRA = fRA(vYt)
    plt.figure(dpi = 300)
    plt.plot(dfBike["Bicycle"], color = 'red')
    plt.plot(np.array(range(2, len(dfBike) + 1)), vRA, color = 'blue')
    plt.legend(labels=["Bicycle Sales", "Running Avg Forecast"])
    plt.show()
    return 

###########################################################
### fPlot16()
def fPlot16(dfBike):
    vYt = dfBike["Bicycle"].values
    vRT, _ = fRT(vYt)
    vRT = np.insert(vRT, 0, [None, None])  # Fix the length
    plt.figure(dpi = 300)
    plt.plot(np.array(range(1, len(dfBike) + 1)), vYt, color='red')  # Adjust x-axis values
    plt.plot(np.array(range(1, len(dfBike) + 1)), vRT, color='blue')  # Adjust x-axis values
    plt.legend(labels=["Bicycle Sales", "Running Trend Forecast"])
    plt.show()
    return 

###########################################################
### fPlot17()
def fPlot17(dfBike):
    vYt = dfBike["Bicycle"].values
    vRT, mBeta = fRT(vYt)
    vRT = np.insert(vRT, 0, [None, None])  # Fix the length
    # Add two new rows of 0 to the beginning of mBeta
    mBeta = np.insert(mBeta, 0, np.zeros((2, 2)), axis=0)

    fig = plt.figure(dpi = 300)
    ax1 = fig.add_subplot(211)
    ax1.plot(np.array(range(1, len(dfBike) + 1)), vYt, color='red')  # Adjust x-axis values
    ax1.plot(np.array(range(1, len(dfBike) + 1)), vRT, color='blue')  # Adjust x-axis values
    ax1.legend(labels=["Bicycle Sales", "Running Trend Forecast"])
    
    ax2 = fig.add_subplot(212)
    n = len(vYt)
    r = np.arange(n)+1 
    a = mBeta[:,0]
    b = mBeta[:,1]  
    width = 0.3
    ax2.bar(r, a, color = 'g', 
        width = width,
        label= r"$a_{t-1}$") 
    ax2.bar(r + width, b, color = 'b', 
        width = width, 
        label= r"$b_{t-1}$") 
    ax2.legend() 

    plt.tight_layout(pad = 1.08)
    plt.show()
    return 

###########################################################
### fPlot18()
def fPlot18(dfBike):
    vYt = dfBike["Bicycle"].values
    vRW = fRW(vYt)
    vRW = np.insert(vRW, 0, [None])  # Fix the length
    vRW_dr, vCt = fRW_Drift(vYt)
    vRW_dr = np.insert(vRW_dr, 0, [None, None])  # Fix the length
    vCt = np.insert(vCt, 0, [0, 0])  # Fix the length

    fig = plt.figure(dpi = 300)
    ax1 = fig.add_subplot(311)
    ax1.plot(np.array(range(1, len(dfBike) + 1)), vYt, color='red')  # Adjust x-axis values
    ax1.plot(np.array(range(1, len(dfBike) + 1)), vRW, color='blue')  # Adjust x-axis values
    ax1.legend(labels=["Bicycle Sales", "Random Walk Forecast"], fontsize="8")
    
    ax2 = fig.add_subplot(312)
    ax2.plot(np.array(range(1, len(dfBike) + 1)), vYt, color='red')  # Adjust x-axis values
    ax2.plot(np.array(range(1, len(dfBike) + 1)), vRW_dr, color='blue')  # Adjust x-axis values
    ax2.legend(labels=["Bicycle Sales", "Random Walk Plus Drift Forecast"], fontsize="8")

    ax3 = fig.add_subplot(313)
    n = len(vYt)
    r = np.arange(n)+1 
    width = 0.5
    ax3.bar(r, vCt, color = 'g', 
        width = width,
        label= r"$c_{t-1}$") 
    ax3.legend(loc="upper left") 

    plt.tight_layout(pad = 1.08)
    plt.show()
    return 

###########################################################
### fPlot19()
def fPlot19(dfBike):
    
    vYt = dfBike["Bicycle"].values
    dAlpha = 0.2
    dBeta = 0.2
    vHW = fHolt_Winters(vYt, dAlpha, dBeta)
    iPeriods = 40
    mWeightsL = np.zeros((iPeriods, iPeriods))
    mWeightsG = np.zeros((iPeriods, iPeriods))
    mWeightsL[0, 0] = 1
    mWeightsG[0, : 2] = np.array([1, -1])
    for T in range(2, iPeriods + 1):
        vWeightsL = np.zeros(iPeriods)
        vWeightsL[: T - 1] = dAlpha * (1 - dAlpha) ** np.array(range(T - 1))
        vWeightsL[T - 1] = (1 - dAlpha) ** (T - 1)
        mWeightsL[T - 1] = vWeightsL
        mWeightsG[T - 1] = dBeta * (mWeightsL[T - 1] - mWeightsL[T - 2]) + (1 - dBeta) * mWeightsG[T - 2]
        
    mWeightsL[-1] + mWeightsG[-1]
    plt.vlines(np.array(range(1, iPeriods + 1)), 0, mWeightsL[-1] + mWeightsG[-1])

    return 

###########################################################
### fPlot20()
def fPlot20(dfBike):
    vYt = dfBike["Bicycle"].values

    
    return 

###########################################################
### fBicyclePredict()



###########################################################
### fBicyclePredict
def fBicyclePredict(dfBike):
    fPlot14(dfBike)
    fPlot15(dfBike)
    fPlot16(dfBike)
    fPlot17(dfBike)
    fPlot18(dfBike)
    return 


###########################################################
### For the umbrella


### fPlot84 page 84 for the umbrella

from matplotlib.ticker import MultipleLocator
def fPlot84(dfUmbrella):
    x_values = np.arange(1, 6, 0.25)
    plt.figure(dpi=300)
    plt.plot(x_values, dfUmbrella["Umbrella Sales"], color='green', marker='o', linestyle='-')

    plt.minorticks_on()

    major_locator = MultipleLocator(1)
    minor_locator = MultipleLocator(0.25)

    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().xaxis.set_minor_locator(minor_locator)

    # Set the vertical scale from 0 to 180
    plt.yticks(np.arange(0, 181, 20))
    # Set non-primary scale labels to empty strings
    plt.yticks(np.arange(0, 181, 5), labels=["" if i % 20 != 0 else str(i) for i in range(0, 181, 5)], minor=True)

    plt.show()

    return

###########################################################
### def fUmbrellaPredict
def fUmbrellaPredict(dfUmbrella):
    
    fPlot84(dfUmbrella)
    
    return 
###########################################################
### fPredict()
def fPredict(vYt, bTune = 1, dAlpha_ES = 0, dAlpha_HW = 0, dBeta_HW = 0):
    
    vRW = fRW(vYt)
    vRA = fRA(vYt)
    vRT, mBeta = fRT(vYt)
    vRW_Drift, vCt = fRW_Drift(vYt)
    
    if bTune == 1:
        vMSE = np.zeros(10)
        vAlpha = np.linspace(0.1, 1, 10)
        vBeta = np.linspace(0.1, 1, 10)
        for i in range(len(vAlpha)):
            vES = fES(vYt, vAlpha[i])
            dME, dMAE, dMAPE, dMSE = fEvaluation(vYt[-11: ], vES[-10: ])
            vMSE[i] = dMSE
        dAlpha_ES = vAlpha[np.argmin(vMSE)]
        
        mMSE = np.zeros((len(vAlpha), len(vBeta)))
        for i in range(len(vAlpha)):
            for j in range(len(vBeta)):
                vHW = fHolt_Winters(vYt, vAlpha[i], vBeta[j])
                dME, dMAE, dMAPE, dMSE = fEvaluation(vYt[-11: ], vHW[-10: ])
                mMSE[i, j] = dMSE
        dAlpha_HW = vAlpha[np.argmin(mMSE) // 10]
        dBeta_HW = vBeta[np.argmin(mMSE) % 10]
        
    vES = fES(vYt, dAlpha_ES)
    vHW = fHolt_Winters(vYt, dAlpha_HW, dBeta_HW)
    
    mPredictions = np.vstack((vRW[-10: ], vRA[-10: ], vRT[-10: ], vRW_Drift[-10: ], vES[-10: ], vHW[-10: ]))
    mEvaluations = np.zeros((len(mPredictions), 4))
    for i in range(len(mPredictions)):
        mEvaluations[i] = fEvaluation(vYt[-11: ], mPredictions[i])
    dfEvaluation = pd.DataFrame(mEvaluations, columns = ['ME', 'MAE', 'MAPE', 'MSE'], index = ['RW', 'RA', 'RT', 'RW_Drift', 'ES', 'HW'])
    
    if bTune == 1:
        return dfEvaluation, dAlpha_ES, dAlpha_HW, dBeta_HW
    else:
        return dfEvaluation

###########################################################
### fTunning_Para()
def fTunning_Para(vYt, sSeason, bMethod, bEva, iCheck):
    
    vAlpha = np.linspace(0.1, 1, 10)
    vBeta = np.linspace(0.1, 1, 10)
    vGamma = np.linspace(0.1, 1, 10)
    mEva = np.zeros(((len(vAlpha), len(vBeta), len(vGamma))))
    for i in range(len(vAlpha)):
        for j in range(len(vBeta)):
            for k in range(len(vGamma)):
                if bMethod == 'Seasonal_HW_Multi':
                    vSHW = fSeasonal_HW_Multi(vYt, sSeason, vAlpha[i], vBeta[j], vGamma[k])
                    if bEva == 'ME':
                        mEva[i, j, k] = fEvaluation(vYt[-iCheck-1: ], vSHW[-iCheck: ])[0]
                    elif bEva == 'MAE':
                        mEva[i, j, k] = fEvaluation(vYt[-iCheck-1: ], vSHW[-iCheck: ])[1]
                    elif bEva == 'MAPE':
                        mEva[i, j, k] = fEvaluation(vYt[-iCheck-1: ], vSHW[-iCheck: ])[2]
                    elif bEva == 'MSE':
                        mEva[i, j, k] = fEvaluation(vYt[-iCheck-1: ], vSHW[-iCheck: ])[3]
                if bMethod == 'Seasonal_HW_Add':
                    vSHW = fSeasonal_HW_Add(vYt, sSeason, vAlpha[i], vBeta[j], vGamma[k])
                    if bEva == 'ME':
                        mEva[i, j, k] = fEvaluation(vYt[-iCheck-1: ], vSHW[-iCheck: ])[0]
                    elif bEva == 'MAE':
                        mEva[i, j, k] = fEvaluation(vYt[-iCheck-1: ], vSHW[-iCheck: ])[1]
                    elif bEva == 'MAPE':
                        mEva[i, j, k] = fEvaluation(vYt[-iCheck-1: ], vSHW[-iCheck: ])[2]
                    elif bEva == 'MSE':
                        mEva[i, j, k] = fEvaluation(vYt[-iCheck-1: ], vSHW[-iCheck: ])[3]
    iIndex = np.argmin(mEva)
    dAlpha_HW = vAlpha[iIndex // 100]
    dBeta_HW = vBeta[iIndex // 10 % 10]
    dGamma_HW = vGamma[iIndex % 10]
    if bMethod == 'Seasonal_HW_Multi':
        vYt_hat = fSeasonal_HW_Multi(vYt, sSeason, dAlpha_HW, dBeta_HW, dGamma_HW)
    elif bMethod == 'Seasonal_HW_Add':
        vYt_hat = fSeasonal_HW_Add(vYt, sSeason, dAlpha_HW, dBeta_HW, dGamma_HW)
    
    return vYt_hat

###########################################################
### fSeasonal_Predcit()
def fSeasonal_Predcit(vYt, sSeason, bEva, iCheck):
    
    vRW_Sea = fSeasonal_RW_Drift(vYt, sSeason)
    vRSR = fRunning_SR(vYt, sSeason)
    
    vSHW_Multi = fTunning_Para(vYt, sSeason, 'Seasonal_HW_Multi', bEva, iCheck)
    vSHW_Add = fTunning_Para(vYt, sSeason, 'Seasonal_HW_Add', bEva, iCheck)
    mPredictions = np.vstack((vRW_Sea[-iCheck: ], vRSR[-iCheck: ], vSHW_Multi[-iCheck: ], vSHW_Add[-12: ]))
    mEvaluations = np.zeros((len(mPredictions), 4))
    for i in range(len(mPredictions)):
        mEvaluations[i] = fEvaluation(vYt[-iCheck-1: ], mPredictions[i])
    dfEvaluation = pd.DataFrame(mEvaluations, columns = ['ME', 'MAE', 'MAPE', 'MSE'], index = ['RW_Sea', 'RSR', 'SHW_Multi', 'SHW_Add'])
    
    return dfEvaluation

###########################################################
### main()
def main():
    
    # Import datasets
    lNames = ['BicycleSales.xlsx', 'GasolineSales1.xlsx', 'GasolineSales2.xlsx', 'Umbrella.xlsx', 'DataAssignment1.xlsx', 'Store Sales.csv']
    dfBike, dfGas1, dfGas2, dfUmbrella, dfData, dfSun = fData(lNames)
    
    # Question (a)
    dfTable1, dfTable2, dfTable3, dfTable4, dfTable5, dfTable7 = fGasPredict(dfGas1, dfGas2)
    fBicyclePredict(dfBike)
    fUmbrellaPredict(dfUmbrella)
    
    # Question (b)
    vYt = dfData['Var3'].values[: 40]
    dfEvaluation, dAlpha_ES, dAlpha_HW, dBeta_HW = fPredict(vYt)
    
    vYt = dfData['Var3'].values
    dfEvaluation = fPredict(vYt, bTune = 0, dAlpha_ES = dAlpha_ES, dAlpha_HW = dAlpha_HW, dBeta_HW = dBeta_HW)
    
    # Question (c)
    vYt = dfUmbrella['Umbrella Sales'].values
    dfEva_Sea = fSeasonal_Predcit(vYt, 'seasonal', 'MAE')

    # Question (d)
    vYt = dfSun['Sunspot'].values
    fPlot1(vYt)
    dfEva_insample = fSeasonal_Predcit(vYt[: -12 * 2], 'monthly', 'MAE')
    dfEva_outsample = fSeasonal_Predcit(vYt, 'monthly', 'MAE')
    
    
###########################################################
### start main
if __name__ == "__main__":
    main()

