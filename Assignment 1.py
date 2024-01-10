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

