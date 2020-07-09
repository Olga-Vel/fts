# Analysis of Financial Time Series, Ruey S. Tsay
# Pag. 12

import pandas as pd
import numpy as np
import scipy.stats as stats
import math
df = pd.read_csv('../data/d-ibm3dx7008.txt', delimiter="\t", header=0)
print(str(df.shape))
print(df.head())
df.rename(str.strip, axis='columns', inplace=True)
ibm=df['rtn'].to_numpy(copy=True)
sibm=ibm*100
print('Minimum: '+str(np.nanmin(sibm)))
print('Maximum: '+str(np.nanmax(sibm)))
print('1. Quartile: '+str(np.nanquantile(sibm, 0.25)))
print('3. Quartile: '+str(np.nanquantile(sibm, 0.75)))
print('Mean: '+str(np.nanmean(sibm)))
print('Median: '+str(np.nanmedian(sibm)))
print('Sum: '+str(np.nansum(sibm)))
print('Variance: '+str(np.nanvar(sibm)))
print('Skewness: '+str(stats.skew(sibm,nan_policy='omit')))
print('Kurtosis: '+str(stats.kurtosis(sibm,nan_policy='omit')))
nsibm=sibm[np.logical_not(np.isnan(sibm))]
print('SE Mean: '+str(stats.sem(nsibm)))
print('Skewness test: '+str(stats.skewtest(nsibm)))
libm=np.log(ibm+1)*100
nlibm=libm[np.logical_not(np.isnan(libm))]
print('T test for 0 mean: '+str(stats.ttest_1samp(nlibm,0)))
print('Normal test: '+str(stats.normaltest(nlibm)))