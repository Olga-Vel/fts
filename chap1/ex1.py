# Analysis of Financial Time Series
# Chap. 1

import pandas as pd
import numpy as np
import scipy.stats as stats
import math

def main():
    df = pd.read_csv('../data/d-3stock.txt', delimiter="\s+", names=['date', 'axp', 'cat', 'sbux'], header=None)
    print(str(df.shape))
    print(df.head())

    # For axp:
    print()
    print('For axp:')
    axp=df['axp'].to_numpy(copy=True)
    calculate(axp)
    
    # For cat:
    print()
    print('For cat:')
    cat=df['cat'].to_numpy(copy=True)
    calculate(cat)
    
    # For sbux:
    print()
    print('For sbux:')
    sbux=df['sbux'].to_numpy(copy=True)
    calculate(sbux)
    
def calculate(data):
    sdata=data*100
    print('Minimum: '+str(np.nanmin(sdata)))
    print('Maximum: '+str(np.nanmax(sdata)))
    print('Mean: '+str(np.nanmean(sdata)))
    print('Median: '+str(np.nanmedian(sdata)))
    print('Variance: '+str(np.nanvar(sdata)))
    print('Skewness: '+str(stats.skew(sdata,nan_policy='omit')))
    print('Kurtosis: '+str(stats.kurtosis(sdata,nan_policy='omit')))
    ldata=np.log(data+1)*100
    print('Minimum: '+str(np.nanmin(ldata)))
    print('Maximum: '+str(np.nanmax(ldata)))
    print('Mean: '+str(np.nanmean(ldata)))
    print('Median: '+str(np.nanmedian(ldata)))
    print('Variance: '+str(np.nanvar(ldata)))
    print('Skewness: '+str(stats.skew(ldata,nan_policy='omit')))
    print('Kurtosis: '+str(stats.kurtosis(ldata,nan_policy='omit')))
    nldata=ldata[np.logical_not(np.isnan(ldata))]
    print('T test for 0 mean: '+str(stats.ttest_1samp(nldata,0)))
    print('Normal test: '+str(stats.normaltest(nldata)))

if __name__=='__main__':
    main()