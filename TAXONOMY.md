# Taxonomy



| Capability     | Option          | Meaning  |
|---------------|--------------------|---------|
| **Training:mode** | none | Does not support training/fitting on data, supervision categories do not apply. |
| | unsupervised | Does not require labeled data or explicit normal/reference data to detect anomalies. |
|  | semi-supervised | Requires explicit reference (normal) data, but no labeled anomalies. |
|  | weakly-supervised | Requires some labeled anomalies, but not exhaustive labelling. |
|  | supervised | Requires extensive or exhaustive labeled anomalies. |
| **Training:update** | yes | Training can be updated, regardless of whether on multiple data instances (according to the data section), or incrementally over time. |
|  | no | If supported, training must be performed in one-go, on a single data instance according to the data section. |
| **Inference:streaming** | yes | Can process new data points incrementally as they arrive, in real time or near-real time, marking anomalies without re-fitting and without any knowledge about the future. Inference is causal: decisions do not depend on future observations. |
|  | no | Must access the full series to mark anomalies on new data points, possibly requiring a re-fitting. Inference is acausal: decisions require future observations (bounded or unbounded) |
| **Inference:dependency** | point | Requires just a single data point of one or more series, according to the data section; each timestamp is processed independently, without temporal context. |
|  | window | Requires a finite local neighborhood of one or more series, according to the data section. |
|  | series| RRequires access to a full series or more full series, according to the data section. |
| **Inference:granularity** | point | Can mark specific timestamps as anomalous across labels (variables). |
|  | point-labels | Can mark specific variables (labels) as anomalous at specific timestamps. |
|  | series | Can only mark entire time series as anomalous, without temporal localization or label-specific marking. |
|  | series-labels | Can mark an entire variable (label) of a multivariate series as anomalous across all timestamps. |
| **Data:dimensionality** (multiple options) | univariate-single | Works on  single univariate time series. |
|  | multivariate-single | Works on single multivariate time series. |
|  | univariate-multi | Works on multiple univariate series. |
|  | multivariate-multi | Works on multiple multivariate series. |
| **Data:sampling** | regular | Each data point is regularly spaced in time. |
|  | irregular | Data points spacing over time is non-uniform. |


Notes:
1. multi-series options assume aligned timestamps unless the model is capable of dealing with irregular sampling.

## Some examples

### MinMaxAnomalyDetector

| Capability     | Value             | Comment |
|---------------|--------------------|---------|
| **Training:mode**  | none | No training required |
| **Training:update**  | no | No training required  |
| **Inference:streaming** | no | Requires to see all the data |
| **Inference:context*** |  point | Needs just one point to mark |
| **Inference:granularity** | label | Can mark per-label |
| **Data:dimensionality** | univariate-single,univariate-multi,multivariate single,multivariate-multi |  Can process both uni-variate and multi-variate as well as sets of them |
| **Data:sampling** | irregular |  Can process irregularly-sampled time series |

### LSTMAnomalyDetector (as semi-supervised)


| Capability     | Value             | Comment |
|---------------|--------------------|---------|
| **Training:mode** | semi-supervised | Thresholds are set based on normal data |
| **Training:update**  | yes | Multiple fit() calls can be performed on several series, thresholds adjust  |
| **Inference:streaming** | yes | Can mark based on the last n samples AND the thresholds |
| **Inference:dependency** |  window | Needs the last n samples |
| **Inference:granularity** | point-labels | Can mark per-label |
| **Data:dimensionality** | univariate-single,multivariate-single |  Can process both uni-variate and multi-variate but no sets of them |
| **Data:sampling** | regular |  Must process evenly spaced datapoints |

### LSTMAnomalyDetector (as unsupervised)
| Capability     | Value             | Comment |
|---------------|--------------------|---------|
| **Training:mode** | unsupervised | Thresholds are set based on data containing both normal and anomalous behevior |
| **Training:update**  | yes | Multiple fit() calls can be performed on several series, thresholds adjust  |
| **Inference:streaming** | no | Even if it could, it still requires accessing all the data to set the threshold |
| **Inference:dependency** |  series | Needs to access all the series to compute thresholds |
| **Inference:granularity** | point-labels | Can mark per-label |
| **Data:dimensionality** | univariate-single,multivariate-single |  Can process both uni-variate and multi-variate but no sets of them |
| **Data:sampling** | regular |  Must process evenly spaced datapoints |

### NHARAnomalyDetector 
| Capability     | Value             | Comment |
|---------------|--------------------|---------|
| **Training:mode** | unsupervised | Thresholds are set based on data containing both normal and anomalous behevior |
| **Training:update**  | no |  Only one fit() phase is supported, thresholds are bound to initial data |
| **Inference:streaming** | no | Needs to access all the data to set the threshold |
| **Inference:dependency** |  series | Needs to access all the data to set the threshold  |
| **Inference:granularity** | point-label | Can mark specific labels |
| **Data:dimensionality** | multivariate-single |  Can process only single multivariate time series |
| **Data:sampling** | regular |  Must process evenly spaced datapoints |
