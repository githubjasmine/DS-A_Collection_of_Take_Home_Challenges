# set index as datetime obj
df.index = pd.to_datetime(df.index)

# plot time series and show the grid
df.plot(grid = True)
plt.show()

# slice the index with a certain year/month/day
df_2012 = df[df.index.year == 2012]

# get the difference of index across two datasets
set_df1_index = set(df1.index)
set_df2_index = set(df2.index)

print("Difference:", set_df1_index - set_df2_index)


# Correlation of Two Time Series:
"""
1. Scatter plot can always be used in visualize the correlation between the two
2. The correlation coefficient is a measure of how much two series vary togetherself.
        - High Corr Coef: strongly vary together
        - Low Corr Coef: vary togther but in a week association
        - High Neg Corr Coef: vary in opposite directions, but still linear relationship
3. Two trending series might have high correlation even if they are actually unrelated.
    In this case, look at the correlation of their "Returns" instead of their actual levels.
"""
# compute the percent change (returns) of the series, and calculate the corr coefficient
returns1 = df1.pct_change()
returns2 = df2.pct_change()
correlation = returns1.corr(returns2)

plt.scatter(returns1, returns2)
plt.show()

# Simple Linear Regression
"""
1. Ordinary Least Squares (OLS):
    minimizes the sum of the squared distances between the data points and the regression line
2. R-Squared: how well the line fit to the data. the maginitude of the corr is the square root of the R^2,
                and the sign of the corr coef is the sign of the slope of the line.
"""
import statsmodels.api as sm
# add a constant column before running regression
df = sm.add_constant(df)
# get the returns of each Series
df['returns1'] = df['series1'].pct_change()
df['returns2'] = df['series2'].pct_change()
# the first row contains NaN, delete it for regression
df = df.dropna()
# run the regression
# sm.OLS(y,x).fit()
results = sm.OLS(df['returns1'], df[['const', 'returns2']]).fit()
print(results.summary())

import numpy as np
np.polyfit(x,y, deg=1)

import pandas as pd
pd.ols(y,x)

from scipy import stats
stats.linregress(x,y)


# AutoCorrelation
"""
- AutoCorrelation: the correlation of a single time series with a lagged copy of itself.
                    can also be called serial correlation
                    always means "lag-one" autocorrelation
- Interpretation of AutoCorrelation:
                    positive auto: trend following
                    negative auto: mean reverting
"""
# Convert daily data to weekly data
df = df.resample(rule='W', how='last') # df.resample(rule="W").last()
returns = df.pct_change()
autocorrelation = returns['ts'].autocorr()
