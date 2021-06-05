# Analysis of Financial Time Series
# Chap. 1

import pandas as pd
import numpy as np
import math

def main():
    df = pd.read_csv('../data/m-ibm3dx7503.txt', delimiter="\s+", names=['date', 'gm', 'vw', 'ew', 'sp'], header=None)
    print(str(df.shape))
    print(df.head())
    
    print()
    print('For sp:')
    sp=df['sp'].to_numpy(copy=True)
    calculate(sp)
    
def calculate(data):
    
    fdata=data+1
    print('Return of $1: $'+str(np.cumprod(fdata)[-1]))
    lfdata=np.log(fdata)
    rlfdata=np.reshape(lfdata,(-1,12))
    arlfdata=np.sum(rlfdata,axis=1)
    aarlfdata=np.mean(arlfdata)
    print(str(np.multiply(np.add(np.exp(aarlfdata).item(),-1),100.0))+' %')


if __name__=='__main__':
    main()