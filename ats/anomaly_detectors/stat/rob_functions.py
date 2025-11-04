# %%
# -*- coding: utf-8 -*-

"""
Created on Wed Apr 23 11:49:40 2025

@author: Fede85
"""

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.linalg import eigh, pinv
from scipy.sparse.linalg import eigs
from scipy.stats import zscore
import matplotlib.pyplot as plt
import random


def extract_date(column_name):
    return pd.to_datetime('_'.join(column_name.split('_')[-3:]), format='%d_%m_%Y')


def plot_random_outliers(vt, mY, GroupAllouts_REGAO, mOuREGAO0):
    chk = GroupAllouts_REGAO.flatten()  # Assicurati sia un array 1D
    col = mOuREGAO0[:, 1]  # colonne
    row = mOuREGAO0[:, 0]  # righe

    selected_vars = random.sample(list(chk), 12)

    fig, axs = plt.subplots(3, 4, figsize=(18, 10))
    axs = axs.flatten()

    for idx, var_index in enumerate(selected_vars):
        axs[idx].plot(vt, mY[var_index, :], 'b')
        outlier_times = row[col == var_index].astype(int)
        axs[idx].plot(vt[outlier_times], mY[var_index, outlier_times], 'r*')
        axs[idx].set_title(f'Variable {var_index}')
        axs[idx].set_xlabel('Time')
        axs[idx].set_ylabel('Value')
        axs[idx].legend()

        axs[idx].tick_params(axis='x', labelsize=8)
        for label in axs[idx].get_xticklabels():
            label.set_rotation(45)
    plt.tight_layout()
    plt.show()


def plot_res(mRes, series_index):

    cn, cd = mRes.shape
    if series_index < 0 or series_index >= cd:
        raise ValueError("Indice serie fuori dal range!")

    x = np.arange(cn)  # Giorni
    y = mRes[:, series_index]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=f"Residuals {series_index}", color="blue")
    plt.xlabel("days")
    plt.ylabel("value")
    plt.title(f"Residuals {series_index}")
    plt.grid(True)
    plt.xticks(fontsize=8)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def plot_ID(oneID, df_numeric):
    if oneID not in df_numeric.index:
        print(f"ID {oneID} non trovato.")
        return

    y_values = df_numeric.loc[oneID]
    x_dates = df_numeric.columns

    plt.figure(figsize=(12, 6))
    plt.plot(x_dates, y_values, linewidth=2)
    plt.title(f"ID {oneID} - Time Series")
    plt.xlabel("Data")
    plt.ylabel("Valori")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fCleanSeries(mY):
    mDY = np.diff(mY, axis=1)
    vnoper = np.sum(mDY != 0, axis=1)
    vsel = np.where(vnoper < 103)[0]
    mY = np.delete(mY, vsel, axis=0)
    vsel2 = np.where(np.all(mY[:, :31] == 0, axis=1))[0]

    mY = np.delete(mY, vsel2, axis=0)
    vsel3 = np.where(np.all(mY[:, -31:] == 0, axis=1))[0]
    mY = np.delete(mY, vsel3, axis=0)
    mDY = np.diff(mY, axis=1)
    cd1, cn1 = mDY.shape
    vsel1 = []

    for i in range(cd1):

        nonzero_indices = np.where(mDY[i, :] != 0)[0]
        positions = np.concatenate(([0], nonzero_indices + 1, [cn1 + 1]))
        gaps = np.diff(positions) - 1
        max_zeros = np.max(gaps)
        if max_zeros > 200:
            vsel1.append(i)
    mY = np.delete(mY, vsel1, axis=0)
    return mY


def longrun_variance(vr, L):

    vr = np.asarray(vr).flatten()
    T = len(vr)
    mu = np.mean(vr)
    vCentered = vr - mu

    gamma0 = np.dot(vCentered, vCentered) / T
    s2 = gamma0

    for k in range(1, L + 1):
        gamma_k = np.dot(vCentered[k:], vCentered[:-k]) / T
        weight = 1 - k / (L + 1)  # Bartlett kernel
        s2 += 2 * weight * gamma_k

    return s2


def fComedian(vxi, vxj):
    sCom = np.median((vxi - np.median(vxi)) * (vxj - np.median(vxj)))
    return sCom


def mad(data, axis=0):
    return np.median(np.abs(data - np.median(data, axis=axis, keepdims=True)), axis=axis)


def fCOMblock_processing_method(mY, block_size):
    n, cd = mY.shape
    mCom = np.full((cd, cd), np.nan)

    medY = np.median(mY, axis=0)
    mY_centered = mY - medY  # broadcasting automatico in NumPy

    num_blocks = int(np.ceil(cd / block_size))

    for bi in range(num_blocks):
        i_start = bi * block_size
        i_end = min((bi + 1) * block_size, cd)
        block_i = mY_centered[:, i_start:i_end]

        for bj in range(bi, num_blocks):
            j_start = bj * block_size
            j_end = min((bj + 1) * block_size, cd)
            block_j = mY_centered[:, j_start:j_end]

            # Broadcasting per ottenere una matrice 3D
            product_array = block_i[:, :, np.newaxis] * block_j[:, np.newaxis, :]
            block_mCom = np.median(product_array, axis=0)

            block_mCom[block_mCom < 1e-9] = 0

            mCom[i_start:i_end, j_start:j_end] = block_mCom

            if bi != bj:
                mCom[j_start:j_end, i_start:i_end] = block_mCom.T

    return mCom


def fRobComedian(mY, cN):
    _, cd = mY.shape

    vMADy = mad(mY, axis=0)
    vMADy[vMADy < 1e-10] = 1e-10

    # Inverse MAD diagonal matrix
    mD = diags(1.0 / vMADy)

    if cd > 3000:
        mCom = fCOMblock_processing_method(mY, 1000)
    else:
        mCom = np.full((cd, cd), np.nan)
        for i in range(cd):
            mCom[i, i] = fComedian(mY[:, i], mY[:, i])
            for j in range(i + 1, cd):
                sCom = fComedian(mY[:, i], mY[:, j])
                mCom[i, j] = sCom
                mCom[j, i] = sCom

    mS = mD @ mCom @ mD.T

    for _ in range(cN):
        if  cd > 3000:
            _ , mE = eigs(mS, k=int(0.05 * mS.shape[0]))
            mE = np.real(mE)
            mQ = np.diag(vMADy) @ mE
            #mZ = (np.linalg.solve(mQ, mY.T)).T
            mZ = np.linalg.lstsq(mQ, mY.T, rcond=None)[0].T
        else:
            _ , mE = eigh(mS)
            mE = np.real(mE)
            mQ = np.diag(vMADy) @ mE
            if np.linalg.cond(mQ) < 1e15:
                #mZ = (np.linalg.solve(mQ, mY.T)).T
                mZ = np.linalg.lstsq(mQ, mY.T, rcond=None)[0].T
            else:
                mZ = (pinv(mQ) @ mY.T).T

        vMADz = mad(mZ, axis=0)
        vMADz[vMADz < 1e-10] = 1e-10
        mGamma = np.diag(vMADz ** 2)
        mS = mQ @ mGamma @ mQ.T

    vm = np.median(mZ, axis=0)

    vMuCOM = mQ @ vm
    mSigCOM = mS

    return vMuCOM, mSigCOM


def fDiagnosticalCheck(mOutliers):
    cn, cd = mOutliers.shape

    # Somma degli outliers per ogni serie
    vOut = np.sum(mOutliers, axis=0)
    TotOutliers = np.sum(vOut)

    # Percentuali di outliers
    PercSeriesOutliers = np.sum(vOut != 0) / cd
    PercSeriesOutliers1 = np.sum(vOut == 1) / cd
    PercSeriesOutliers2 = np.sum(vOut == 2) / cd
    PercSeriesOutliers3 = np.sum(vOut > 2) / cd

    # Gruppi di outliers
    GroupAllouts = np.where(vOut > 0)[0]
    GroupOut1 = np.where(vOut == 1)[0]
    GroupOut2 = np.where(vOut == 2)[0]
    GroupOut3 = np.where(vOut > 2)[0]

    # Outliers che appaiono più di una volta
    GroupOutk = np.where(vOut > 1)[0]
    mOutliersk = mOutliers[:, GroupOutk]

    # Gruppo di outliers settimanali
    GroupOutweek = []
    tlaps = 6
    for i in range(cn - tlaps):
        vS1 = np.sum(mOutliersk[i:i + tlaps, :], axis=0)
        check = np.where(vS1 > 1)[0]
        if check.size > 0:
            GroupOutweek.extend(GroupOutk[check])

    GroupOutweek = np.unique(GroupOutweek)

    return TotOutliers, PercSeriesOutliers, PercSeriesOutliers1, PercSeriesOutliers2, PercSeriesOutliers3, GroupAllouts, GroupOut1, GroupOut2, GroupOut3, GroupOutweek


def fFindThr(ccount, coefMask=1e-5, BinWidth=0.2, PLT=0, fitng='Pareto'):
    cN = len(ccount)

    counts, edges = np.histogram(ccount, bins=np.arange(min(ccount), max(ccount) + BinWidth, BinWidth), density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    peak_idx = np.argmax(counts)
    tail_mask = np.zeros_like(counts, dtype=bool)
    tail_mask[peak_idx:] = counts[peak_idx:] > np.max(counts) * coefMask
    tail_mask[bin_centers < 0] = False

    tail_centers = bin_centers[tail_mask]
    tail_log_counts = np.log10(counts[tail_mask] + np.finfo(float).eps)

    if len(tail_centers) == 0:
        raise ValueError("La coda è vuota. Parametri troppo restrittivi?")

    if fitng.lower() == 'linear':
        fit_length = int(len(tail_centers) * 0.8)
        X_fit = tail_centers[:fit_length]
        y_fit = tail_log_counts[:fit_length]
        p = np.polyfit(X_fit, y_fit, 1)
        predicted_log_counts = np.polyval(p, tail_centers)
    else:
        # Pareto fitting
        fit_length = int(len(tail_centers) * 0.9)
        tail_data_for_fit = tail_centers[:fit_length]
        xm_hat = np.min(tail_data_for_fit)
        alpha_hat = len(tail_data_for_fit) / np.sum(np.log(tail_data_for_fit / xm_hat))

        valid_mask = tail_centers >= xm_hat
        tail_centers = tail_centers[valid_mask]
        tail_log_counts = tail_log_counts[valid_mask]

        pareto_pdf = (alpha_hat * xm_hat**alpha_hat) / (tail_centers**(alpha_hat + 1))
        pareto_pdf[pareto_pdf <= 0] = np.finfo(float).eps
        offset = np.mean(tail_log_counts[:fit_length] - np.log10(pareto_pdf[:fit_length]))
        predicted_log_counts = np.log10(pareto_pdf) + offset

    residuals = tail_log_counts - predicted_log_counts
    residuals_std = np.std(residuals[:fit_length])
    threshold_factor = 3
    residual_threshold = np.mean(residuals[:fit_length]) + threshold_factor * residuals_std

    threshold_idx = fit_length
    for i in range(fit_length, len(residuals)):
        if np.abs(residuals[i]) > residual_threshold:
            threshold_idx = i
            break

    if threshold_idx < len(tail_centers):
        distance_threshold = tail_centers[threshold_idx]
    else:
        distance_threshold = tail_centers[-1]

    ccount_sorted = np.sort(ccount)
    prc = np.where(np.abs(ccount_sorted - distance_threshold) == np.min(np.abs(ccount_sorted - distance_threshold)))[0][0] / cN

    if PLT:
        plt.figure()
        plt.plot(tail_centers, tail_log_counts, 'o', label='Tail Log Counts')
        plt.plot(tail_centers, predicted_log_counts, 'r-', label='Fitted')
        plt.axvline(distance_threshold, color='r', linestyle='--',
                    label=f'Q({prc:.4f}) = {distance_threshold:.2f}')
        plt.title(r'Log Tail Fitting and threshold')
        plt.xlabel('Distance')
        plt.ylabel('Log Tail')
        plt.legend()

        plt.figure()
        plt.plot(bin_centers, np.log10(counts + np.finfo(float).eps))
        plt.title("Log10(PDF)")

        plt.figure()
        plt.hist(ccount, bins=np.arange(min(ccount), max(ccount) + BinWidth, BinWidth), density=True)
        plt.title("PDF Histogram")

        plt.show()

    return distance_threshold, prc, tail_centers, predicted_log_counts, tail_log_counts, tail_mask


def fTrimmedCUSUM(vy, sigma2, c, alpha):
    cn = len(vy)
    C2max = 0
    ctau = 0
    vv = np.zeros(cn)

    for t in range(1, cn):
        # Segmenta i dati
        seg1 = vy[:t]
        seg2 = vy[t:]

        # Ordina i segmenti
        seg1_sorted = np.sort(seg1)
        seg2_sorted = np.sort(seg2)

        # Calcola gli indici di trimming
        k1 = max(1, int(np.floor(alpha * t)))        # Numero di elementi da rimuovere dal primo segmento
        k2 = max(1, int(np.floor(alpha * (cn - t)))) # Numero di elementi da rimuovere dal secondo segmento

        # Applica la media trimmata se ci sono abbastanza dati rimasti
        if t > k1 and (cn - t) > k2:
            trim_seg1 = seg1_sorted[k1:]   # Rimuove i k1 valori più piccoli
            trim_seg2 = seg2_sorted[k2:]   # Rimuove i k2 valori più piccoli
        else:
            trim_seg1 = seg1_sorted
            trim_seg2 = seg2_sorted

        # Assicurati che i segmenti non siano vuoti
        if len(trim_seg1) == 0 or len(trim_seg2) == 0:
            continue

        # Calcola le medie trimmate
        cy1 = np.mean(trim_seg1)
        cy2 = np.mean(trim_seg2)

        # Calcola la statistica C2
        C2 = ((t * (cn - t)) / cn) * (cy1 - cy2)**2

        # Aggiorna la statistica massima
        if C2 > C2max:
            C2max = C2
            ctau = t

    # Regola di decisione
    if C2max / sigma2 > c:
        r1 = ctau
        r2 = C2max
        vv[ctau:] = 1  # Segna tutti i punti dopo il changepoint come "cambiati"
    else:
        r1 = None
        r2 = C2max
        vv = None

    return r1, r2, vv


def lts_regression(X, y, h=None, num_iters=500, random_state=42):

    n, p = X.shape
    rng = np.random.default_rng(random_state)
    if h is None:
        h = int(np.floor(0.75 * n))

    vbest_b = None
    best_sse = np.inf

    for _ in range(num_iters):
        try:
            # Selezione casuale iniziale di p punti
            subset = rng.choice(n, p, replace=False)
            X_sub = X[subset, :]
            y_sub = y[subset]

            # Stima iniziale
            vb_temp = np.linalg.solve(X_sub.T @ X_sub, X_sub.T @ y_sub)

            # Calcolo residui
            vres0 = np.abs(y - X @ vb_temp)
            idx = np.argsort(vres0)

            # Subset con h osservazioni
            X_trimmed = X[idx[:h], :]
            y_trimmed = y[idx[:h]]

            # Stima finale sul subset
            vb_final = np.linalg.solve(X_trimmed.T @ X_trimmed, X_trimmed.T @ y_trimmed)
            sse = np.sum((y_trimmed - X_trimmed @ vb_final) ** 2)

            if sse < best_sse:
                best_sse = sse
                vbest_b = vb_final

        except np.linalg.LinAlgError:
            continue

    if vbest_b is None:
        raise ValueError("Non è stato possibile trovare una soluzione valida")

    # Calcolo residui robusti
    y_pred = X @ vbest_b
    vres = (y - y_pred)#/np.std(y - y_pred, ddof=0)
    # print(f"vbest_b: {vbest_b}")
    return vres, vbest_b


def lts_regression2(X, y, h=None, num_iters=500, random_state=None):

    n, p = X.shape
    rng = np.random.default_rng(random_state)
    if h is None:
        h = int(np.floor(0.75 * n))

    vbest_b = None
    best_sse = np.inf

    for _ in range(num_iters):
        try:
            # Selezione casuale iniziale di p punti
            subset = rng.choice(n, p, replace=False)
            X_sub = X[subset, :]
            y_sub = y[subset]

            # Stima iniziale
            vb_temp = np.linalg.solve(X_sub.T @ X_sub, X_sub.T @ y_sub)

            # Calcolo residui
            vres0 = np.abs(y - X @ vb_temp)
            idx = np.argsort(vres0)

            # Subset con h osservazioni
            X_trimmed = X[idx[:h], :]
            y_trimmed = y[idx[:h]]

            # Stima finale sul subset
            vb_final = np.linalg.solve(X_trimmed.T @ X_trimmed, X_trimmed.T @ y_trimmed)
            sse = np.sum((y_trimmed - X_trimmed @ vb_final) ** 2)

            if sse < best_sse:
                best_sse = sse
                vbest_b = vb_final

        except np.linalg.LinAlgError:
            continue

    if vbest_b is None:
        # raise ValueError("Non è stato possibile trovare una soluzione valida")
        vres = np.full(n, np.nan)
        vbest_b=np.full(p, np.nan)
    else:
        vres = y - X @ vbest_b
    # print(f"vbest_b: {vbest_b}")
    return vres, vbest_b


def fRobResiduals(mY,fq= 2 * np.pi / 30, fw=2 * np.pi / 7,TREND=2):
    cd, cn0 = mY.shape
    fw = np.atleast_1d(fw)
    fq = np.atleast_1d(fq)

    t = np.arange(1, cn0 + 1)
    if TREND==2:
        mTrend=  np.column_stack([
            np.ones((cn0, 1)),
            t,
            t**2 ])
    elif TREND==1:
        mTrend=  np.column_stack([
            np.ones((cn0, 1)),
            t])
    elif TREND==0:
        mTrend = np.ones((cn0, 1))
    else:
        raise ValueError('Values of TREND not allowed, allowed values are 0, 1, 2') 

    # Matrice dei regressori
    if len(fw) == 0 and len(fq) == 0:
        mX = mTrend
    elif len(fw) == 0:
        mX = np.column_stack([
            mTrend,
            np.cos(fq * t[:, None]),
            np.sin(fq * t[:, None]),
        ])
    elif len(fq) == 0:
        mX = np.column_stack([
            mTrend,
            np.cos(fw * t[:, None]),
            np.sin(fw * t[:, None]),
        ])
    else:
        mX = np.column_stack([
            mTrend,
            np.cos(fw * t[:, None]),
            np.sin(fw * t[:, None]),
            np.cos(fq * t[:, None]),
            np.sin(fq * t[:, None]),
        ])

    mRes = np.full((cn0, cd), np.nan)

    for i in range(cd):
        # Regressione robusta: residui sulla riga i-esima
        res, brob = lts_regression(mX, mY[i, :].T)
        #print(f"Iterazione: {i}, brob: {brob}")
        mRes[:, i] = res
        #print(f"Robust residuals {i}")
        #print(i)
    # Differenze temporali dei residui
    mDRes = np.diff(mRes, axis=0)

    return mRes, mDRes


def fLagmatrix(vr, lag):
    vr = np.asarray(vr)
    n = len(vr)
    if n <= lag:
        raise ValueError("La lunghezza del vettore deve essere maggiore del lag")

    mRlag = np.zeros((n - lag, lag))
    for i in range(lag):
        mRlag[:, lag - 1 - i] = vr[i : n - lag + i]
    return mRlag


def compute_HARresiduals(vr):
    mRlag0 = fLagmatrix(vr, 30)
    mRlag = np.column_stack([ #np.ones((len(vr)-30, 1)),
        mRlag0[:, 0],
        np.mean(mRlag0[:, 0:7], axis=1),
        np.mean(mRlag0[:, 0:30], axis=1)
    ])
    _, brob = lts_regression(mRlag, vr[30:])
    rhat = mRlag @ brob
    e = vr[30:] - rhat
    return e**2


def process_column(i, mRes, mDRes):
    vr = mRes[:, i]
    vDr = mDRes[:, i]
    return compute_HARresiduals(vr), compute_HARresiduals(vDr)


def fARcorsi(mRes, mDRes):
    cn0, cd = mRes.shape
    cn, _ = mDRes.shape
    mResSqr = np.empty((cn0 - 30, cd))
    mResSqrD= np.empty((cn - 30, cd))

    for i in range(cd):
        vr=mRes[:, i]
        mResSqr[:, i] = compute_HARresiduals(vr)
        vDr=mDRes[:, i]
        mResSqrD[:, i] = compute_HARresiduals(vDr)   

    '''
    with parallel_backend('loky'):  # Default, può anche essere 'threading'
        results = Parallel(n_jobs=-1)(
            delayed(process_column)(i, mRes, mDRes) for i in range(cd)
        )

    mResSqr = np.column_stack([r[0] for r in results])
    mResSqrD = np.column_stack([r[1] for r in results])
   '''

    #mResSqr = np.array(mResSqr).T  # Trasposta per allineamento
    mResSqrZ = zscore(mResSqr, axis=0)

    #ccount_AR = mResSqrZ.flatten()
    ccount_AR = np.sort(mResSqrZ.ravel())
    threshold_AR, prc, _, _, _, _ = fFindThr(ccount_AR, 1e-5, 0.2, 0,'Linear')
    outliers_REG = mResSqrZ > threshold_AR
    row, col = np.where(outliers_REG)
    mOuREGAO = np.column_stack((row + 30, col))
 
    #mResSqrD = Parallel(n_jobs=-1)(
    #    delayed(compute_HARresiduals)(mDRes[:, i]) for i in range(cd)
    #)
    #mResSqrD = [compute_HARresiduals(mDRes[:, i]) for i in range(cd)]
    #mResSqrD = np.array(mResSqrD).T
    mResSqrZD = zscore(mResSqrD, axis=0)
    #ccount_ARD = mResSqrZD.flatten()
    ccount_ARD = np.sort(mResSqrZD.ravel())
    threshold_ARD, _, _, _, _, _ = fFindThr(ccount_ARD, 1e-5, 0.2, 0)

    outliers_REGD = mResSqrZD > threshold_AR
    row, col = np.where(outliers_REGD)
    mOuREGLS = np.column_stack((row + 31, col))

    combined = np.vstack([mOuREGAO, mOuREGLS])
    unique_rows = np.unique(combined, axis=0)
    sort_idx = np.argsort(unique_rows[:, 1])
    mOuREG = unique_rows[sort_idx, :]

    return mOuREG, mOuREGAO, mOuREGLS

def analyze_level_shifts(mOuREG, mRes, mY, mX , vt ):
    col = mOuREG[:, 1]
    mResREG = mRes[:, col]
    vC1 = np.full(len(col), np.nan)

    for i in range(len(col)):
        s2 = np.var(mResREG[:, i])
        _, c1, _ = fTrimmedCUSUM(mResREG[:, i], s2, 0, 0.01)
        vC1[i] = c1

    th1 = np.quantile(vC1, 0.8)
    vLSO_REG = []
    vBetaREG = []

    for i in range(len(col)):
        s2 = np.var(mResREG[:, i])
        r2, _, vv = fTrimmedCUSUM(mResREG[:, i], s2, th1, 0.01)
        if r2:
            _, vBeta = lts_regression(np.column_stack([mX, vv]), mY[col[i], :])
            vBetaREG.append(vBeta)
            vLSO_REG.append([r2, col[i], vBeta[-1]])

    plt.figure(figsize=(12, 6))
    plt.hist(vt[mOuREG[:, 0]], bins=1000)
    plt.xticks(fontsize=8)
    plt.xticks(rotation=45)
    plt.show

    plt.figure(figsize=(12, 6))
    vLSO_REGnp=np.array(vLSO_REG)
    vLSneg =  vLSO_REGnp[:, 0].copy()  # Colonna 1 in Python è colonna 0
    mask = vLSO_REGnp[:, 2] <= 0      # Colonna 3 in Python è colonna 2
    vLSneg = vLSneg[mask]          # Mantieni solo i valori dove mask è False
    plt.hist(vt[vLSneg.astype(int)], bins=1000)
    plt.xticks(fontsize=8)
    plt.xticks(rotation=45)
    plt.show

    plt.figure(figsize=(12, 6))
    vLSpos = vLSO_REGnp[:, 0].copy()  # Colonna 1 in Python è colonna 0
    mask = vLSO_REGnp[:, 2] >= 0      # Colonna 3 in Python è colonna 2
    vLSpos = vLSpos[mask]          # Mantieni solo i valori dove mask è False
    plt.hist(vt[vLSpos.astype(int)], bins=1000)
    plt.xticks(fontsize=8)
    plt.xticks(rotation=45)
    plt.show

    plt.figure(figsize=(12, 6))
    mOuREG_notidf = mOuREG.copy()
    mask = np.isin(mOuREG_notidf[:, 1], vLSO_REGnp[:, 1])
    mOuREG_notidf = mOuREG_notidf[~mask]
    plt.hist(vt[mOuREG_notidf[:, 0].astype(int)], bins=1000)
    plt.xticks(fontsize=8)
    plt.xticks(rotation=45)
    plt.show
    return vLSO_REG, vBetaREG 


def analyze_AOs(mOuREG, mRes, mY, mX  ):
    cn0 = mX.shape[0]
    n_vars = mOuREG.shape[0]
    p = mX.shape[1]
    vBetaREG = np.full((p + 2, n_vars), np.nan)
    vTestTao = np.full(n_vars, np.nan)
    vTestTlso = np.full(n_vars, np.nan)
    mYz = zscore(mY, axis=1)

    for i in range(n_vars):
        shift_idx = int(mOuREG[i, 0])
        col_idx = int(mOuREG[i, 1]) - 1  # da 1-based a 0-based

        if shift_idx > 1:
            vvAO = np.zeros((cn0, 1))
            vvAO[shift_idx-1] = 1  # -1 per shift 1-based
            vvLSO = np.zeros((cn0, 1))
            vvLSO[shift_idx-1:] = 1  # -1 per shift 1-based

            mXaug = np.column_stack([mX, vvAO, vvLSO])
            y_target = mYz[col_idx, :].T
            vr, vBeta  = lts_regression2(mXaug, y_target)

            vBetaREG[:, i] = vBeta 
            s2 = mad(vr, axis=0) 
            invXtX = np.linalg.inv(mXaug.T @ mXaug)
            invXtX_diag = np.diag(invXtX)
            vTestTao[i] = vBeta[-2] / np.sqrt(s2 * invXtX_diag[-2])
            vTestTlso[i] = vBeta[-1] / np.sqrt(s2 * invXtX_diag[-1])
 

    #vMask = np.abs(vBetaREG[-1, :]) > 0.1
    vMaskAO= np.abs(vTestTao) > 2.576
    vMaskLSO= np.abs(vTestTlso) < 2.576
    vMask= vMaskAO & vMaskLSO
    mOuREGAO0 = mOuREG[vMask, :]
    mOuREGLSO0 = mOuREG[vMaskLSO, :] 
    return vBetaREG, vTestTao, mOuREGAO0, mOuREGLSO0


def classify_outlier(vr, t0, window=5):

    before = np.mean(vr[max(0, t0 - window):t0])
    after = np.mean(vr[t0 + 1:t0 + 1 + window])
    current = vr[t0]

    # se il punto è distante da prima e dopo
    if abs(current - before) > 2*np.std(vr) and abs(current - after) > 2*np.std(vr):
        return 'AO'  # spike singolo

    # se da t0 in poi c'è uno shift medio
    if abs(after - before) > 2*np.std(vr):
        return 'LSO'

    return 'None'


def fPerioSeasfq(vy, cseas):

    cn = len(vy)
    vomega = 2 * np.pi * np.arange(cn) / cn  # Frequenze di Fourier

    vPerio = (np.abs(np.fft.fft(vy - np.mean(vy)))**2) / (cn * 2 * np.pi)
    vdiff1 = (vomega - cseas[0])**2
    cindex = np.argmin(vdiff1)
    idx1 = max(cindex - 2, 0)
    idx2 = min(cindex + 3, len(vPerio))
    cSeasPerio = np.max(vPerio[idx1:idx2])
    vdiff2 = (vomega - cseas[1])**2
    cindex2 = np.argmin(vdiff2)
    idx3 = max(cindex2 - 2, 0)
    idx4 = min(cindex2 + 3, len(vPerio))
    cSeasPerio2 = np.max(vPerio[idx3:idx4])
    cZeroPerio = np.max(vPerio[0:5])

    vfreq = np.array([cZeroPerio, cSeasPerio, cSeasPerio2])
    return vfreq


def fCOM(mRes,mDRes):

    #COM estimator
    cn, cd = mDRes.shape
    ccij = np.zeros_like(mDRes)
    vmu_com2, mCov_com2 = fRobComedian(mDRes, 2)
    for i in range(cn):
        diffz = mDRes[i, :] - vmu_com2           # vmu_com2 è vettore di lunghezza cd
        ccij[i, :] = diffz**2 / np.diag(mCov_com2)

    mCZDcom = zscore(ccij, axis=0)
    ccountD_com = np.sort(mCZDcom.ravel())
    thresholdD_com, prcD_com, tail_centersD_com, predicted_log_countsD_com, tail_log_countsD_com, tail_maskD = \
        fFindThr(ccountD_com, 1e-5, 0.2, 0)
    outliers_com = mCZDcom > thresholdD_com
    row, col = np.where(outliers_com)
    mOuCOMLS = np.column_stack([row + 1, col])  
    cn, cd = mRes.shape
    ccij = np.zeros_like(mRes)
    # 2. stima robusta della media e covarianza
    vmu_com2, mCov_com2 = fRobComedian(mRes, 2)
    # 3. calcolo del contributo per variabile tramite Mahalanobis distances
    for i in range(cn):
        diffz = mRes[i, :] - vmu_com2           # vmu_com2 è vettore di lunghezza cd
        ccij[i, :] = diffz**2 / np.diag(mCov_com2)

    mCZcom = zscore(ccij, axis=0)
    ccount_com = np.sort(mCZcom.ravel())
    threshold_com, prc_com, tail_centers_com, predicted_log_counts_com, tail_log_counts_com, tail_mask = \
        fFindThr(ccount_com, 1e-5, 0.2, 0,'Linear')

    outliers_com = mCZcom > threshold_com
    row, col = np.where(outliers_com)
    mOuCOMAO = np.column_stack([row , col])  

    # merge AOs and LSOs

    df1 = pd.DataFrame(mOuCOMLS, columns=['row', 'col'])
    df2 = pd.DataFrame(mOuCOMAO, columns=['row', 'col'])
    combined = pd.concat([df1, df2], ignore_index=True)
    unique_rows = combined.drop_duplicates()
    mOuCOMall_df = unique_rows.sort_values(by='col').reset_index(drop=True)
    mOuCOMall = mOuCOMall_df.values

    return mOuCOMall, mOuCOMAO, mOuCOMLS

