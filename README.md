# Anomalies in Time Series (ATS)

This repository provides clean, ready to use interfaces for common anomaly detection models on time series data, as well as novel implementations.

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


## Supported methods

This is a list of the various methodologies available as of today, together with their class/module names.

---

### MinMaxAnomalyDetector

A trivial anomaly detector based on the minimum and maximum values of the time series.

**class:** : `anomaly_detectors.naive.minmax.MinMaxAnomalyDetector `

---

### IFSOMAnomalyDetector

Anomaly detector based on Isolation Forest and Self-organizing maps [3]

- **class:** `anomaly_detectors.stats.ifsom.IFSOMAnomalyDetector `

---

### COMAnomalyDetector

Robust anomaly detection technique based on the covariance matrix [2]

**class:** `anomaly_detectors.stats.robust.COMAnomalyDetector`

---

### NHARAnomalyDetector

Robust anomaly detection technique based on non-linear regression via neural networks and residuals modeling [2]

- **class:** `anomaly_detectors.stats.robust. NHARAnomalyDetector `

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
