# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 16:27:50 2025

@author: Fede85
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import multivariate_normal
import pandas as pd
from .rob_functions   import (fRobResiduals,fCOM, fARcorsi)
from .NAR_functions import fNHAR


def simulate_gaussian(cd, cn0):

    veig = np.random.rand(cd)
    mA = 25*np.random.randn(cd, cd)
    mSigma = mA @ np.diag(veig) @ mA.T

    vmu = np.zeros(cd)
    mEps1 = multivariate_normal.rvs(mean=vmu, cov=mSigma, size=cn0).T

    return mEps1, mSigma


def fSimCycModel(cn, dsigma2_eps, dsigma2_zeta, dsigma2_eta,
                 iTrend, control_points_days, csigma_cyc,
                 control_points_values, ve):
 
    vbeta0 = np.zeros(1000 + cn)
    vmu0 = np.zeros(1000 + cn)

    veps = np.sqrt(dsigma2_eps) * np.random.randn(cn + 1000)
    vzeta = np.sqrt(dsigma2_zeta) * np.random.randn(cn + 1000)

    if iTrend == 1:
        for t in range(cn + 1000):
            vbeta0[t] = np.sum(vzeta[:t+1])   
        for t in range(cn + 1000):
            vmu0[t] = np.sum(veps[:t+1]) + np.sum(vbeta0[:t+1])

    vmu = vmu0[1000:]

    cs_month = CubicSpline(control_points_days, csigma_cyc * control_points_values)

    vcyc = np.full(cn, np.nan)
    veta = np.sqrt(dsigma2_eta) * np.random.randn(cn)

    day_in_month = 0 % 30
    vcyc[0] = cs_month(day_in_month) + veta[0]

    for i in range(1, cn):
        day_in_month = (i % 30)
        vcyc[i] = cs_month(day_in_month) + veta[i]

    vy = vmu + vcyc + ve
    return vy 


def simulate_synData(cd, cn0):

    mY = np.empty((cn0  , cd))
    mEps1, mSigma  =simulate_gaussian(cd, cn0)
    for i in range(cd):
        dsigma2_eta= 0.001*np.random.rand() 
        cn=cn0
        dsigma2_eps= 0.001*np.random.rand() 
        dsigma2_zeta= 0.001*np.random.rand() 
        iTrend=1
        control_points_days=[0, 14 ,15, 30]
        csigma_cyc=100*np.random.rand() 
        control_points_values= np.repeat(np.random.rand(2), 2)
        ve=mEps1[i,:]
        vy = fSimCycModel(cn, dsigma2_eps, dsigma2_zeta, dsigma2_eta, 
                         iTrend, control_points_days, csigma_cyc, 
                         control_points_values, ve)
        mY[:,i] = np.asarray(vy).flatten()
    return mY


def fmSparse(cd, cn, cnj, cdj, vSdiag, outlier='addout', tim=5, seas=7, cphi=1):
    """
    Crea una matrice "sparsa" con outlier di diversi tipi.

    cd      : numero di righe
    cn      : numero di colonne
    cnj     : indici temporali degli outlier (lista o array)
    cdj     : indici delle serie (righe) da modificare (lista o array)
    vSdiag  : varianze (array di lunghezza cd)
    outlier : tipo di outlier ('addout', 'levshif', 'slopchange', 'seasout', 'innout')
    seas    : periodo stagionale per 'seasout'
    cphi    : parametro per 'seasout' o 'innout'
    tim     : scala dell'outlier
    """

    mSparse = np.zeros((cd, cn))

    cdj = np.atleast_1d(cdj).astype(int)
    cnj = np.atleast_1d(cnj).astype(int)

    if outlier.lower() == 'addout':
        # additive outlier
        for n in range(len(cnj)):
            mSparse[cdj, cnj[n]] = tim * np.sqrt(vSdiag[cdj])

    elif outlier.lower() == 'levshif':
        # level shift
        for d in cdj:
            mSparse[d, cnj[0]:cn] = tim * np.sqrt(vSdiag[d])

    elif outlier.lower() == 'slopchange':
        for d in cdj:
            mSparse[d, cnj[0]:cn] = np.arange(1, cn - cnj[0] + 2) * ((tim / 100) * np.sqrt(vSdiag[d]))

    elif outlier.lower() == 'seasout':
        vseas = np.zeros(cn - cnj[0])
        for i in range(0, len(vseas), seas):
            vseas[i] = cphi
        for d in cdj:
            mSparse[d, cnj[0]:cn] = vseas * (tim * np.sqrt(vSdiag[d]))

    elif outlier.lower() == 'innout':
        vma = np.array([cphi**i for i in range(cn - cnj[0])])
        for d in cdj:
            mSparse[d, cnj[0]:cn] = vma * (tim * np.sqrt(vSdiag[d]))

    return mSparse


def fSimContaminatedSeries(cn,cd,prct=0.4,cnj= np.array([80, 100, 120]),outlier='addout',tim=5,seas=7, repr=False):
    '''

    Parameters
    ----------
    cn :  number of observations (minimum 121)
    cd : number of time series   (minimum 10)
    prct : percentage of contaminated series. The default is 0.4.
    cnj : Location in time of the outliers. The default is np.array([80, 100, 120]).
    outlier : outliers typology (for instance, 'addout' and 'levshif') The default is 'addout'.

    Returns
    -------
    mY : Matrix of contaminated time series.
    mOtlrs : Matrix of outlier location 
    '''
    if cn<121:
        raise ValueError('we need minimum 121 observation')
    if cd<10:
        raise ValueError('we need minimum 10 time series')

    if repr:
        np.random.seed(0)

    cnj = np.atleast_1d(cnj).astype(int)
    cN = int(np.floor(cd * prct))  # numero di serie temporali da modificare
    cdj = np.arange(0, cN )  
    mYraw = simulate_synData(cd, cn)
    vSdiag = np.var(mYraw, axis=0, ddof=1)  
    mSparse = fmSparse(cd, cn, cnj, cdj, vSdiag, outlier,tim,seas)
    mY=mYraw.T + mSparse
    mOtlrs = np.array([[c, d] for d in cdj for c in cnj])
    return mY, mOtlrs


def fDemo(mY,mOtlrs,strctmdl=True,fq= 2 * np.pi / 30, fw=2 * np.pi / 7,TREND=2, COM=True, HAR=True, NHAR=True):

    # Compute robust residuals
    if strctmdl:
        mRes, mDRes = fRobResiduals(mY, fq, fw, TREND)
    else:
        mRes=mY.T
        mDRes=np.diff(mY.T, axis=0)


    df_mOtlrs = pd.DataFrame(mOtlrs, columns=['row', 'col'])    

    # Detection via COM methodology
    if COM:
        mOuCOMall, mOuCOMAO, mOuCOMLS  =  fCOM(mRes,mDRes)
        df_mOuCOMall = pd.DataFrame(mOuCOMall, columns=['row', 'col'])
        merged = pd.merge(df_mOuCOMall, df_mOtlrs, on=['row', 'col'], how='left', indicator=True)
        ia = merged['_merge'] == 'both'  # Righe comuni tra mOuCOMall e mOuREG
        num_righe_comuni = ia.sum()
        num_righe_non_comuni = (merged['_merge'] == 'left_only').sum()
        print(f"OUTPUT_1: The {round(100*num_righe_comuni/ df_mOtlrs.shape[0])}% of outliers have been deteceted by the COM estimator")
        print(f"OUTPUT_1: The number of false positive is {num_righe_non_comuni} for the COM estimator")
    else:
        df_mOuCOMall=[]

    # Detection via the HAR methodology
    if  HAR:
        mOuREG, mOuREGAO, mOuREGLS = fARcorsi(mRes, mDRes)
        df_mOuREG = pd.DataFrame(mOuREG, columns=['row', 'col'])
        merged = pd.merge(df_mOuREG, df_mOtlrs, on=['row', 'col'], how='left', indicator=True)
        ia = merged['_merge'] == 'both'  # Righe comuni tra mOuCOMall e mOuREG
        num_righe_comuni = ia.sum()
        num_righe_non_comuni = (merged['_merge'] == 'left_only').sum()
        print(f"OUTPUT_2: The {round(100*num_righe_comuni/ df_mOtlrs.shape[0])}% of outliers have been deteceted by the HAR methodology")
        print(f"OUTPUT_2: The number of false positive is {num_righe_non_comuni} for the HAR methodology")
    else:
        df_mOuREG=[]

    # Detection via the NHAR methodology
    if NHAR:
        mOuNAR, mOuNARAO,  mOuNARLS = fNHAR(mRes , mDRes, epochs=50)       
        df_mOuNAR = pd.DataFrame(mOuNAR, columns=['row', 'col']) 
        merged = pd.merge(df_mOuNAR, df_mOtlrs, on=['row', 'col'], how='left', indicator=True)
        ia = merged['_merge'] == 'both'  # Righe comuni tra mOuCOMall e mOuREG
        num_righe_comuni = ia.sum()
        num_righe_non_comuni = (merged['_merge'] == 'left_only').sum()
        print(f"OUTPUT_3: The {round(100*num_righe_comuni/ df_mOtlrs.shape[0])}% of outliers have been deteceted by the NHAR methodology")
        print(f"OUTPUT_3: The number of false positive is {(num_righe_non_comuni)} for the NHAR methodology")
    else:
        df_mOuNAR=[]

    if not (COM or HAR or NHAR):
        raise ValueError('No method has been selected')

    return df_mOuCOMall, df_mOuREG, df_mOuNAR


def detect_outliers_on_data(mY, fq=2 * np.pi / 30, fw=2 * np.pi / 7, TREND=2,
                            COM=True, HAR=True, NHAR=True):
    """
    Applica i diversi metodi di detection su un dataset di serie temporali (senza validazione).
    """
    # Calcolo residui robusti
    mRes, mDRes = fRobResiduals(mY, fq, fw, TREND)

    results = {}

    if COM:
        mOuCOMall, _, _ = fCOM(mRes, mDRes)
        results['COM'] = pd.DataFrame(mOuCOMall, columns=['row', 'col'])

    if HAR:
        mOuREG, _, _ = fARcorsi(mRes, mDRes)
        results['HAR'] = pd.DataFrame(mOuREG, columns=['row', 'col'])

    if NHAR:
        mOuNAR, _, _ = fNHAR(mRes, mDRes, epochs=50)
        results['NHAR'] = pd.DataFrame(mOuNAR, columns=['row', 'col'])

    return results


def generate_contaminated_dataframe(cn, cd, prct=0.4, cnj=np.array([80, 100, 120]),
                                    outlier='addout', tim=5, seas=7,
                                    start_time='1970-01-01 00:00:00', freq='H', repr=False):
    """
    Wrapper around fSimContaminatedSeries() that returns a pandas DataFrame
    with datetime index (hourly intervals starting from '1970-01-01').

    Parameters
    ----------
    cn : int
        Number of time observations (minimum 121)
    cd : int
        Number of time series (minimum 10)
    prct : float, optional
        Percentage of contaminated series. Default is 0.4.
    cnj : array-like, optional
        Time indices of outliers. Default is [80, 100, 120].
    outlier : str, optional
        Type of outlier ('addout', 'levshif', etc.). Default is 'addout'.
    tim : float, optional
        Outlier scale parameter. Default is 5.
    seas : int, optional
        Seasonal period for 'seasout'. Default is 7.
    start_time : str or datetime, optional
        Start datetime for the index. Default is '1970-01-01 00:00:00'.
    freq : str, optional
        Sampling frequency (default 'H' for hourly).

    Returns
    -------
    df_data : pandas.DataFrame
        DataFrame with datetime index (named "timestamp") and each column a time series.
    df_outliers : pandas.DataFrame
        DataFrame with outlier locations (timestamp and series_index)
    """

    # Generate the contaminated series using the provided function
    mY, mOtlrs = fSimContaminatedSeries(
        cn=cn, cd=cd, prct=prct, cnj=cnj, outlier=outlier, tim=tim, seas=seas, repr=repr
    )

    # Transpose because fSimContaminatedSeries returns (cd, cn)
    mY = mY.T

    # Create datetime index
    time_index = pd.date_range(start=start_time, periods=cn, freq=freq)
    time_index.name = "timestamp"

    # Build the main DataFrame
    df_data = pd.DataFrame(mY, index=time_index, columns=[f"series_{i}" for i in range(cd)])

    # Build outlier DataFrame with corresponding timestamps
    df_outliers = pd.DataFrame(mOtlrs, columns=["time_index", "series_index"])
    df_outliers["timestamp"] = time_index[df_outliers["time_index"]]
    df_outliers = df_outliers[["timestamp", "series_index"]]

    return df_data, df_outliers

