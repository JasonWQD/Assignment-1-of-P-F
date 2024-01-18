import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dfUmbrella = pd.read_excel("D:/BDS2/block 3/Predication & Forecasting/Assignment 1/Umbrella.xlsx")


##############################page 84
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

fPlot84(dfUmbrella)






#####################page 86
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(dfUmbrella["Umbrella Sales"], trend='add', seasonal='add', seasonal_periods=4, use_boxcox=False)

hw_model = model.fit(optimized=True, remove_bias=False)
forecast = hw_model.forecast(steps=8)
print(forecast)



def fPlot86(dfUmbrella, forecast_values):
    x_known = np.arange(1, 6, 0.25)
    x_forecast = np.arange(6, 8, 0.25)

    plt.figure(dpi=300)

    plt.plot(x_known, dfUmbrella["Umbrella Sales"], color='green', marker='o', linestyle='-')

    plt.plot(x_forecast, forecast_values, color='blue', marker='o', linestyle='-')

   
    plt.axvline(x=6, color='black', linestyle='--')

    plt.minorticks_on()

    major_locator = MultipleLocator(1)
    minor_locator = MultipleLocator(0.25)

    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().xaxis.set_minor_locator(minor_locator)

    
    plt.yticks(np.arange(0, 181, 20))
    
    plt.yticks(np.arange(0, 181, 5), labels=["" if i % 20 != 0 else str(i) for i in range(0, 181, 5)], minor=True)

    plt.show()

    return 

forecast_values = forecast
fPlot86(dfUmbrella, forecast_values)

#######################
###########Strange here: dLt only will not include the first two points and the last two points, but dHt has all of the points
import statsmodels.api as sm

result = sm.tsa.seasonal_decompose(dfUmbrella["Umbrella Sales"].values, model='additive', period=4)
dLt = pd.Series(result.trend).dropna()
dHt = result.seasonal
print(dLt)
print(dHt)

###################
########first plot at page 85
def fPlot85(dfUmbrella):

    dfUmbrella_trimmed = dfUmbrella.iloc[2:-2]

    x_values = np.arange(1.5, 5.5, 0.25)
    plt.figure(dpi=300)

    plt.plot(x_values, dfUmbrella_trimmed["Umbrella Sales"], color='green', marker='o', linestyle='-')

    plt.plot(x_values, dLt, color='red', linestyle='-', linewidth=2)

    plt.minorticks_on()

    major_locator = MultipleLocator(1)
    minor_locator = MultipleLocator(0.25)

    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().xaxis.set_minor_locator(minor_locator)

    plt.yticks(np.arange(0, 181, 20))
    
    plt.yticks(np.arange(0, 181, 5), labels=["" if i % 20 != 0 else str(i) for i in range(0, 181, 5)], minor=True)

    plt.show()
    return

fPlot85(dfUmbrella)


#######################
##########second plot at page 85, and this is same as the slides
def fPlot85_G2(dfUmbrella):
    x_values = np.arange(1, 6, 0.25)
    plt.figure(dpi=300)


    plt.plot(x_values, dHt, color='blue', linestyle='-', marker='o')

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    plt.minorticks_on()

    major_locator = MultipleLocator(1)
    minor_locator = MultipleLocator(0.25)

    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().xaxis.set_minor_locator(minor_locator)

    plt.yticks([-25,0,25])

    plt.show()
    return

fPlot85_G2(dfUmbrella)


##########################
######## third plot at page 85, I only keep the points without the first and the last two points, but the result is also somehow strange
def fPlot85_G3(dfUmbrella):

    dfUmbrella_trimmed = dfUmbrella.iloc[2:-2]
    dHt_trimmed = dHt[2:-2]

    x_values = np.arange(1.5, 5.5, 0.25)
    plt.figure(dpi=300)

    Grows = dfUmbrella_trimmed["Umbrella Sales"] - dLt - dHt_trimmed 

    plt.plot(x_values, Grows, color='black', marker='o', linestyle='')

    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    plt.vlines(x_values, ymin=0, ymax=Grows, color='black', linestyle='-', linewidth=1)

    plt.minorticks_on()

    major_locator = MultipleLocator(1)
    minor_locator = MultipleLocator(0.25)

    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().xaxis.set_minor_locator(minor_locator)

    plt.yticks([-5, 0, 5])

    plt.show()

fPlot85_G3(dfUmbrella)
