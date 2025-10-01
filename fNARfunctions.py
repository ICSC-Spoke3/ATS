# -*- coding: utf-8 -*-
"""
Created on Tue May 20 19:57:16 2025

@author: Fede85
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tensorflow.keras.callbacks import EarlyStopping
from keras.saving import register_keras_serializable
from scipy.stats import zscore


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

def fRobResiduals(mY):
    cd, cn0 = mY.shape

    fq = 2 * np.pi / 30
    fw = 2 * np.pi / 7
    t = np.arange(1, cn0 + 1)

    # Matrice dei regressori
    mX = np.column_stack([
        np.ones(cn0),
        t,
        t**2,
        np.cos(fw * t),
        np.sin(fw * t),
        np.cos(fq * t),
        np.sin(fq * t),
    ])

    mRes = np.full((cn0, cd), np.nan)

    for i in range(cd):
        # Regressione robusta: residui sulla riga i-esima
        res, brob = lts_regression(mX, mY[i, :].T)
        #print(f"Iterazione: {i}, brob: {brob}")
        mRes[:, i] = res
        print(f"Robust residuals {i}")
        #print(i)
    # Differenze temporali dei residui
    mDRes = np.diff(mRes, axis=0)

    return mRes, mDRes


def fFindThr(ccount, coefMask=1e-5, BinWidth=0.2, PLT=False, fitng='Pareto'):
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




def fLagmatrix(vr, lag):
    vr = np.asarray(vr)
    n = len(vr)
    if n <= lag:
        raise ValueError("La lunghezza del vettore deve essere maggiore del lag")

    mRlag = np.zeros((n - lag, lag))
    for i in range(lag):
        mRlag[:, lag - 1 - i] = vr[i : n - lag + i]
    return mRlag

def create_dataset(vr):
    mRlag0 = fLagmatrix(vr, 30)
    X=np.column_stack([ 
        mRlag0[:, 0],
        np.mean(mRlag0[:, 0:7], axis=1),
        np.mean(mRlag0[:, 0:30], axis=1)
    ])
    Y=vr[30:]

    return np.array(X), np.array(Y).reshape(-1, 1)



def trimmed_mse_percent(y_true, y_pred):
    residuals = tf.abs(y_true - y_pred)
    sorted_residuals = tf.sort(residuals)
    n = tf.size(residuals)
    q=0.75
    h = tf.cast(tf.floor(q * tf.cast(n, tf.float32)), tf.int32)   
    trimmed = sorted_residuals[:h]
    return tf.reduce_mean(tf.square(trimmed))

@register_keras_serializable()
def custom_loss(y_true, y_pred):
    return trimmed_mse_percent(y_true, y_pred)
    
def build_model(input_dim):
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=custom_loss)
    return model




early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

###  A L L    D A T A S E T

def predict_one_series(i, y, model):
    print(f"Processing series {i}")
    X, Y = create_dataset(y)
    preds = model.predict(X, verbose=0) 
    err2 = np.square(Y.ravel() - preds.ravel())
    '''
    plt.figure()
    plt.plot(Y.ravel)
    plt.plot(preds.ravel)
    plt.show
    '''
    return i, preds, err2

    
def train_and_predict(mRes, mRes50K, epochs=50):
    n_rows, n_cols = mRes.shape
    pred_len = n_rows - 30
    #print('start training')

    _, d_0 = mRes50K.shape
    X_all, Y_all = [], []
    for idx in range(d_0):
        y = mRes50K[:, idx]
        X, Y = create_dataset(y)
        X_all.append(X)
        Y_all.append(Y)

    X_all = np.vstack(X_all)
    Y_all = np.vstack(Y_all)
    del mRes50K
    #print(f"Train shape: X={X_all.shape}, Y={Y_all.shape}")

    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model = build_model(input_dim=X_all.shape[1])
    model.fit(X_all, Y_all, epochs=epochs, batch_size=512, verbose=0, callbacks=[early_stop])
    #512
    mPred = np.full((pred_len, n_cols), np.nan)
    mErr2 = np.full((pred_len, n_cols), np.nan)
    #print('start prediction')
    '''
    results = Parallel(n_jobs=-1)(
        delayed(predict_one_series)(i, mRes[:, i], model)
        for i in range(n_cols)
    )
    '''
    for i in range(n_cols):
        y = mRes[:, i]
        #print(f"Predicting series {i}")
        X, Y = create_dataset(y)
        preds = model.predict(X, verbose=0)
        mPred[:, i] = preds.ravel()
        mErr2[:, i] = np.square(Y.ravel() - preds.ravel())
    ''' 
    print('end prediction')
    for i, preds, err2 in results:
        print(f'collecting results {i}')
        mPred[:, i] = preds.ravel()
        mErr2[:, i] = err2
    '''    
    return mPred, mErr2, model


## T R A I N I N G    F O R    U N I V A R I A T E    T I M E    S E R I E S


def train_single_series(i, y, epochs, callbacks=None):
    print(f"Serie {i + 1}")
    X, Y = create_dataset(y)
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model = build_model(input_dim=3)
    model.fit(X, Y, epochs=epochs, batch_size=32, verbose=0,callbacks=[early_stop])
    preds = model.predict(X, verbose=0)
    err2 = np.square(Y - preds).ravel()

    return i, preds.ravel(), err2

def train_on_matrix_joblib(mRes , epochs=50, n_jobs=-1):
    cn, cd = mRes.shape
    mErr2 = np.empty((cn - 30, cd))
    mPred = np.empty((cn - 30, cd))

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_single_series)(i, mRes[:, i], epochs,callbacks=[early_stop])
        for i in range(cd)
    )

    for i, preds, err2 in results:
        mPred[:, i] = preds
        mErr2[:, i] = err2

    return mPred, mErr2


def fNHAR(mRes , mDRes, epochs=50):

    _,mErr2,model=train_and_predict(mRes , mRes, epochs=50) 


    mErr2Z = zscore(mErr2, axis=0)
    ccount_NAR = np.sort(mErr2Z.ravel())
    threshold_NAR, _, _, _, _, _ = fFindThr(ccount_NAR, 1e-5, 0.2, 0,'Linear')
    outliers_NAR = mErr2Z > threshold_NAR
    row, col = np.where(outliers_NAR)
    mOuNARAO = np.column_stack((row + 30, col))

    _,mDErr2,modelD=train_and_predict(mDRes, mDRes, epochs=50)


    mDErr2Z = zscore(mDErr2, axis=0)
    ccountD_NAR = np.sort(mDErr2Z.ravel())
    thresholdD_NAR, _, _, _, _, _ = fFindThr(ccountD_NAR, 1e-5, 0.2, 0)
    outliersD_NAR = mDErr2Z > thresholdD_NAR
    row, col = np.where(outliersD_NAR)
    mOuNARLS = np.column_stack((row + 31, col))


    combined = np.vstack([mOuNARAO, mOuNARLS])
    unique_rows = np.unique(combined, axis=0)
    sort_idx = np.argsort(unique_rows[:, 1])
    mOuNAR = unique_rows[sort_idx, :]
    
    return mOuNAR, mOuNARAO,  mOuNARLS