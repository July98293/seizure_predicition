# Patient-Specific Seizure Prediction on CHB-MIT

A reproducible EEG **seizure prediction** pipeline on the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/). The model learns to separate **preictal** EEG (the minutes leading up to a seizure) from **interictal** EEG (seizure-free periods) — and is evaluated at the seizure level with a leakage-free cross-validation scheme.

## What it does

Continuous EEG is bandpass-filtered (0.5–50 Hz) with a **60 Hz** mains notch (CHB-MIT was recorded in the US) and cut into 10-second windows. Each window is reduced to a feature vector across the 17 channels common to all recordings:

- relative band power (δ, θ, α, β, γ)
- spectral edge frequency (95%)
- line length
- Hjorth parameters (activity, mobility, complexity)
- statistical moments (std, skew, kurtosis, RMS)

An **XGBoost** classifier (RandomForest fallback) is trained to distinguish preictal from interictal windows.

### Labeling

For a seizure with onset `T`:

```
preictal  = [T − SPH − PREICTAL_LEN,  T − SPH]      (default: 35 → 5 min before onset)
interictal = windows from seizure-free recordings
```

`SPH` (Seizure Prediction Horizon) is an intervention gap excluded right before onset, so an alarm leaves time to act.

### Avoiding leakage

Windows from the same recording then leak across train and test, and the model simply memorizes the recording so every window from a given EDF file is entirely in train *or* entirely in test, never both.

### From windows to alarms (firing power)

A 5% per-window false-positive rate becomes ~18 false alarms/hour at 360 windows/hour. So the per-window probabilities are smoothed with a moving average (firing power), and an alarm fires only when it stays above threshold, followed by a refractory period. Isolated false positives cancel out. A seizure is counted as **predicted** if any alarm fires within its preictal window.

Defaults: 10-minute moving average, threshold 0.6, 30-minute refractory period.

## Results (patient `chb01`)

| Metric | Value |
|---|---|
| Seizures predicted | **6 / 7** (sensitivity 0.86) |
| False-alarm rate | **0.12 / hour** (8 interictal hours) |
| Window-level pooled AUC | 0.96 |

Two honest caveats:

- The single missed seizure (`chb01_21`) its onset is at 327 s, so the 35-minute preictal segment falls almost entirely before the file begins, leaving only 2 usable windows. No classifier can fire without signal.
- The 0.12/hour rate rests on a **single** false alarm across 8 hours; its confidence interval is wide. It would require more data

The naive window-level reading (sensitivity 0.83, specificity 0.95, ~19 false alarms/hour) is reported only to show why per-window metrics mislead and why the firing-power step matters.

## Usage

Open the notebook in Google Colab and run all cells. It downloads patient `chb01` directly from PhysioNet, builds the windows, trains, and evaluates.

```python
PATIENT          = "chb01"   # change to run chb02 … chb24
WINDOW_SEC       = 10
PREICTAL_MIN     = 30
SPH_MIN          = 5
INTERICTAL_FILES = 8
NOTCH_HZ         = 60
```

Outputs a per-run summary (CSV + JSON) with sensitivity, false-alarm rate, and configuration.

## Limitations & next steps

This is a **single-patient proof of concept**. Seizure prediction is strongly patient-specific, so results must not be generalized without more data, a chort wide evaluation and a more deep model.

## Data & references

- CHB-MIT Scalp EEG Database, PhysioNet: https://physionet.org/content/chbmit/1.0.0/
- Shoeb, A. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD thesis, MIT.
- Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.

The CHB-MIT database is distributed under the PhysioNet open-access license; cite the sources above if you use it.
