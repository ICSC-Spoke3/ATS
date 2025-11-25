# Anomalies in Time Series (ATS)

This repository provides clean, ready to use interfaces for common anomaly detection techniques in time series data, as well as novel implementations.

It also offers a synthetic benchmark to be used more as testbed for understanding each method pros and cons (since benchmarking the unknown is mostly useless [1])

Example notebooks are as well provides, and can be found int he root of the repository.


## Supported methods

This is a list of the various methodologies available, together with their capabilities and class/module names. Each also states if it can work on single series (univariate or multivariate), lists of series (univariate or multivariate), or both.

---

`MinMaxAnomalyDetector`

- **Description:** trivial anomaly detector based on the minimum and maximum values of the time series
- **Type:** unsupervised
- **Works on:** single series (uv or mv)
- **Real time:** no
- **Module:** `anomaly_detectors.naive.minmax`

---

`IFSOMAnomalyDetector`

- **Description:** anomaly detector based on Isolation Forest and Self-organizing maps
- **Type:** unsupervised
- **Works on:** lists of series (uv)
- **Real time:** no
- **Module:** `anomaly_detectors.stats.ifsom`

---

`COMAnomalyDetector`

- **Description:** Robust anomaly detection technique based on the covariance matrix [2]
- **Type:** unsupervised
- **Works on:** single series (mv) or lists of series (uv)
- **Real time:** no
- **Module:** `anomaly_detectors.stats.robust`

---

`NHARAnomalyDetector`

- **Description:** Robust anomaly detection technique based on non-linear regression via neural networks and residuals modeling [2]
- **Type:** unsupervised
- **Works on:** single series (mv) or lists of series (uv)
- **Real time:** no
- **Module:** `anomaly_detectors.stats.robust`

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

[1] **Robust anomaly detection for time series data in sensor-based critical systems**. Stefano Alberto Russo https://arts.units.it/handle/11368/3107341 

[2] **Anomaly Detection in High-Dimensional Bank Account Balances via Robust Methods**. Federico Maddanu, Tommaso Proietti, Riccardo Crupi https://arxiv.org/abs/2511.11143 

[3] **Navigating AGN variability with self-organizing maps**. Ylenia Maruccia, Demetra De Cicco, Stefano Cavuoti, Giuseppe Riccio, Paula Sánchez-Sáez, Maurizio Paolillo, Noemi Lery Borrelli, Riccardo Crupi, Massimo Brescia https://www.aanda.org/articles/aa/pdf/2025/07/aa53866-25.pdf
