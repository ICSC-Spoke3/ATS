# Anomalies in Time Series (ATS)

This repository provides a modular framework with clean, well-defined interfaces for time series anomaly detection. It includes both classical methods and novel example implementations, making it easy to experiment with, extend, and compare approaches.

Each model has well-defined `capabilites`, following a specific [taxonomy](TAXONOMY.md) which allows to clearly state what models can and cannot do, from an operational standpoint. For example:

```
>>> LSTMAnomalyDetector.capabilities
{
  "training": {
    "mode": "semi-supervised",
    "update": False
  },
  "inference": {
    "streaming": true,
    "dependency": "window",
    "granularity": "point-labels"
  },
  "data": {
    "dimensionality": ["univariate-single",
                       "multivariate-single"],
    "sampling": "regular"
  }
}
```

It also offers a synthetic data generator for testing purposes, and an evaluator focusing more on model strengths and weaknesses rather than absolute performance.

Example notebooks are provided as well, and can be found in the root of the repository.


## Example models

This is a non-exhaustive list of the models implemented in the framework, together with their module/class names.

---

### MinMaxAnomalyDetector

A trivial anomaly detector based on the minimum and maximum values of the time series.

**class:** : `anomaly_detectors.naive.minmax.MinMaxAnomalyDetector `

---

### PeriodicAverageAnomalyDetector

A simple statistical method based on computing average values over a given period.

**class:** : `anomaly_detectors.stat.periodic_average.PeriodicAverageAnomalyDetector `

---

### IFSOMAnomalyDetector

Anomaly detector based on Isolation Forest and Self-organizing maps [1]

- **class:** `anomaly_detectors.ml.ifsom.IFSOMAnomalyDetector `

---

### COMAnomalyDetector

A robust anomaly detector based on the Comedian (COM) estimator [2]

**class:** `anomaly_detectors.stats.robust.COMAnomalyDetector`

---

### NHARAnomalyDetector

A robust anomaly detection technique based on non-linear regression via neural networks and residuals modelling [2]

- **class:** `anomaly_detectors.stats.robust.HARAnomalyDetector `

---


### LSTMAnomalyDetector

An anomaly detector based on capturing time series dynamics via a LSTM neural network [3]

- **class:** `anomaly_detectors.dl.lstm.LSTMAnomalyDetector `

---

## Usage

Setup (with virtualenv):

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Run tests:

    python -m unittest discover
    
Optionally:

    pip install jupyterlab==4.4.3
    jupyter lab --port 9999


## References

[1] **Anomaly Detection in High-Dimensional Bank Account Balances via Robust Methods**. Federico Maddanu, Tommaso Proietti, Riccardo Crupi https://arxiv.org/abs/2511.11143 

[2] **Navigating AGN variability with self-organizing maps**. Ylenia Maruccia, Demetra De Cicco, Stefano Cavuoti, Giuseppe Riccio, Paula Sánchez-Sáez, Maurizio Paolillo, Noemi Lery Borrelli, Riccardo Crupi, Massimo Brescia https://www.aanda.org/articles/aa/pdf/2025/07/aa53866-25.pdf

[3] **Timeseria: an object-oriented time series processing library**. Stefano Alberto Russo, Giuliano Taffoni, Luca Bortolussi, https://www.sciencedirect.com/science/article/pii/S2352711025000032
