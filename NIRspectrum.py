# ======================================================================
# NIR OM/TC — ROUND-2 REVISION ANALYSES (Reviewer 1, second round)
# ======================================================================
#
# One script that produces every new number, figure and supplementary
# table requested in the second review round:
#
#   R1-1  Precision (PPV) and confusion matrices on the original split;
#         classifier performance over repeated stratified 70/30 splits
#         (mean, SD, 2.5–97.5 percentile range)               -> S6, S7
#   R1-2  Regression with SG, SG+SNV, SG+MSC and SG 1st derivative;
#         R2, RMSE, bias, RPD, RPIQ; repeated random splits    -> S6, S8
#   R1-3  Simple baselines (mean absorbance; best single wavelength);
#         LASSO coefficients and PLSR VIP scores           -> Figure 4
#   R1-4  2 x 2 table of OM > 4% vs TC > 2.32% labels, Cohen's kappa -> S5
#   R1-6  Selected hyperparameters of all models (original split)  -> S4
#   R1-7  ROC curves as one two-panel figure (a) OM, (b) TC   -> Figure 3
#
# All tables are also written to Supplementary_Tables_S4_S8.docx, and the
# values for every [XX] placeholder in the manuscript are printed and saved
# in numbers_for_manuscript.txt.
#
# The original-split models use the same split, preprocessing and search
# grids as the published Tables 1-3:
#   classification: 70/30 split stratified by the OM class, seed 42
#   regression:     70/30 unstratified random split, seed 42
#
# ALL MODELS, INCLUDING THE 1D-CNNs AND FUSED LASSO, ARE EVALUATED OVER
# 100 REPEATED SPLITS (N_REPEATS = N_SLOW = 100).
#
# HOW TO RUN ON A LAPTOP (Windows, macOS or Linux, Python 3.9-3.12):
#   1. Install the packages once (in a terminal / Anaconda Prompt):
#        pip install numpy pandas scipy scikit-learn imbalanced-learn xgboost
#        pip install cvxpy clarabel torch matplotlib python-docx
#   2. Put this script and NIRagri.csv in the same folder, open a terminal
#      in that folder and run:
#        python round2_analyses_100repeats.py NIRagri.csv
#   Quick test first (2 repeats, about 10-20 minutes):
#        python round2_analyses_100repeats.py NIRagri.csv 2 2
#      then delete the folder round2_outputs before the full run.
#
# SAVING TIME: progress is written to CSV after every repeat, and a
# restarted run skips everything already in round2_outputs. If you still
# have round2_outputs/S7_raw_repeated_classification.csv and
# round2_outputs/S8_raw_repeated_regression.csv from the previous run
# (100 repeats for the fast models, 20 for the 1D-CNN and fused LASSO),
# put them in a folder named round2_outputs next to this script: the same
# split seeds are used, so only repeats 21-100 of the 1D-CNN and fused
# LASSO are computed. You can stop the run (Ctrl+C) and start it again at
# any time; it resumes where it stopped. Keep the laptop plugged in and
# switch off sleep mode for long runs.
# ======================================================================

import os, re, sys, time, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold,
                                     GridSearchCV, ParameterGrid)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score,
                             precision_score, roc_auc_score, roc_curve,
                             confusion_matrix, cohen_kappa_score,
                             r2_score, mean_squared_error, mean_absolute_error)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import cvxpy as cp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# When the code is pasted into a Colab/Jupyter cell, edit these three lines.
# When it is run as a script (!python round2_analyses.py FILE N N_SLOW),
# the command-line values replace them.
FILE = "NIRagri.csv"
N_REPEATS = 100   # repeats for all models
N_SLOW = 100      # repeats for the 1D-CNN and fused LASSO (same as N_REPEATS)

IN_NOTEBOOK = "ipykernel" in sys.modules
_args = [] if IN_NOTEBOOK else sys.argv[1:]
if len(_args) > 0:
    FILE = _args[0]
if len(_args) > 1:
    N_REPEATS = int(_args[1])
if len(_args) > 2:
    N_SLOW = int(_args[2])
else:
    N_SLOW = min(N_SLOW, N_REPEATS)
OUT = "round2_outputs"
os.makedirs(OUT, exist_ok=True)

SEED = 42
SG_WINDOW, SG_POLY = 11, 2
OM_CUT, TC_CUT = 4.00, 2.32

def seed_everything(s):
    np.random.seed(s); random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| repeats:", N_REPEATS, "| slow-model repeats:", N_SLOW)

def out(name):
    return os.path.join(OUT, name)

# ======================================================================
# 1. DATA
# ======================================================================
df = pd.read_csv(FILE)
non_spectral = ["Sample number", "OM", "TC"]
spectral_cols = [c for c in df.columns if c not in non_spectral]

def extract_wavelength(name):
    n = re.findall(r"[-+]?\d*\.\d+|\d+", str(name))
    return float(n[-1]) if n else np.nan

wavelengths = np.array([extract_wavelength(c) for c in spectral_cols])
if not np.any(np.isnan(wavelengths)):
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    spectral_cols = [spectral_cols[i] for i in order]
else:
    wavelengths = np.arange(len(spectral_cols), dtype=float)

X_raw = df[spectral_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
OM = pd.to_numeric(df["OM"], errors="coerce").to_numpy(float)
TC = pd.to_numeric(df["TC"], errors="coerce").to_numpy(float)
mask = np.isfinite(OM) & np.isfinite(TC) & np.all(np.isfinite(X_raw), axis=1)
X_raw, OM, TC = X_raw[mask], OM[mask], TC[mask]
y_OM = (OM > OM_CUT).astype(int)
y_TC = (TC > TC_CUT).astype(int)
idx = np.arange(len(OM))
print(f"Samples: {len(OM)} | wavelengths: {X_raw.shape[1]} | "
      f"OM positives: {y_OM.sum()} | TC positives: {y_TC.sum()}")

# ======================================================================
# 2. PREPROCESSING
# ======================================================================
def sg(X):
    return savgol_filter(X, SG_WINDOW, SG_POLY, deriv=0, axis=1, mode="interp")

def sg_d1(X):
    return savgol_filter(X, SG_WINDOW, SG_POLY, deriv=1, axis=1, mode="interp")

def snv(X):
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)

def msc(Xtr, Xte):
    """MSC with the reference (mean) spectrum estimated from training data only."""
    ref = Xtr.mean(0)
    def correct(X):
        Z = np.empty_like(X)
        for i, s in enumerate(X):
            b, a = np.polyfit(ref, s, 1)
            Z[i] = (s - a) / b
        return Z
    return correct(Xtr), correct(Xte)

X_SG = sg(X_raw)

def preprocess(name, tr, te):
    """Return (X_train, X_test) for a preprocessing method; no test leakage."""
    if name == "SG":
        return X_SG[tr], X_SG[te]
    if name == "SG + SNV":
        Z = snv(X_SG); return Z[tr], Z[te]
    if name == "SG 1st derivative":
        Z = sg_d1(X_raw); return Z[tr], Z[te]
    if name == "SG + MSC":
        return msc(X_SG[tr], X_SG[te])
    raise ValueError(name)

PREPROCESSING = ["SG", "SG + SNV", "SG + MSC", "SG 1st derivative"]

# ======================================================================
# 3. METRICS
# ======================================================================
def reg_metrics(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    rmse = np.sqrt(mean_squared_error(y, p))
    q1, q3 = np.percentile(y, [25, 75])
    return {"R2": r2_score(y, p), "RMSE": rmse, "MAE": mean_absolute_error(y, p),
            "Bias": float(np.mean(p - y)), "RPD": np.std(y, ddof=1) / rmse,
            "RPIQ": (q3 - q1) / rmse}

def cls_metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, pred),
            "Balanced accuracy": balanced_accuracy_score(y, pred),
            "Sensitivity": recall_score(y, pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "Precision (PPV)": precision_score(y, pred, zero_division=0),
            "AUC": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}

def fmt_summary(df_, group_cols, metric_cols, digits=3):
    """mean ± SD [2.5–97.5 percentile] per group."""
    g = df_.groupby(group_cols, sort=False)[metric_cols]
    m, s = g.mean(), g.std()
    lo, hi = g.quantile(0.025), g.quantile(0.975)
    n = g.size().rename("Repeats")
    res = pd.DataFrame(index=m.index)
    for c in metric_cols:
        res[c] = (m[c].map(lambda v: f"{v:.{digits}f}") + " ± " +
                  s[c].map(lambda v: f"{v:.{digits}f}") + " [" +
                  lo[c].map(lambda v: f"{v:.{digits}f}") + "–" +
                  hi[c].map(lambda v: f"{v:.{digits}f}") + "]")
    return res.join(n).reset_index()

# ======================================================================
# 4. CLASSIFIERS
# ======================================================================
class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS regression on 0/1 labels; class 1 if predicted score >= 0.5."""
    def __init__(self, n_components=2):
        self.n_components = n_components
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False).fit(X, y)
        return self
    def decision_function(self, X):
        return self.pls_.predict(X).ravel()
    def predict(self, X):
        return (self.decision_function(X) >= 0.5).astype(int)

def smote():
    return SMOTE(random_state=SEED, k_neighbors=3)

# Full grids = those used for the published Tables 2-3
FULL_GRIDS = {
    "LASSO logistic": {"m__C": np.logspace(-3, 3, 20)},
    "PLS-DA": {"m__n_components": [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]},
    "SVM": {"m__C": [0.1, 1, 10, 100], "m__gamma": ["scale", 0.001, 0.01, 0.1]},
    "Random forest": {"m__n_estimators": [200, 500], "m__max_depth": [None, 5, 10],
                      "m__min_samples_leaf": [1, 2, 4], "m__max_features": ["sqrt", 0.5]},
    "XGBoost": {"m__n_estimators": [100, 300], "m__learning_rate": [0.03, 0.10],
                "m__max_depth": [2, 3], "m__min_child_weight": [1, 5],
                "m__subsample": [0.8], "m__colsample_bytree": [0.5, 0.8],
                "m__reg_alpha": [0, 0.1, 1], "m__reg_lambda": [0.1, 1, 10]},
}
# Reduced grids for RF and XGBoost in the repeated evaluation (run time)
REPEAT_GRIDS = dict(FULL_GRIDS)
REPEAT_GRIDS["Random forest"] = {"m__n_estimators": [300], "m__max_depth": [None, 5],
                                 "m__min_samples_leaf": [1, 4], "m__max_features": ["sqrt"]}
REPEAT_GRIDS["XGBoost"] = {"m__n_estimators": [300], "m__learning_rate": [0.03, 0.10],
                           "m__max_depth": [2, 3], "m__min_child_weight": [1, 5],
                           "m__subsample": [0.8], "m__colsample_bytree": [0.5],
                           "m__reg_alpha": [0, 1], "m__reg_lambda": [1, 10]}

def classifier_pipeline(name):
    if name == "LASSO logistic":
        return ImbPipeline([("s", StandardScaler()), ("sm", smote()),
                            ("m", LogisticRegression(penalty="l1", solver="liblinear",
                                                     max_iter=20000, random_state=SEED))])
    if name == "PLS-DA":
        return ImbPipeline([("s", StandardScaler()), ("sm", smote()), ("m", PLSDA())])
    if name == "SVM":
        return ImbPipeline([("s", StandardScaler()), ("sm", smote()),
                            ("m", SVC(kernel="rbf", probability=True, random_state=SEED))])
    if name == "Random forest":   # no scaling needed
        return ImbPipeline([("sm", smote()),
                            ("m", RandomForestClassifier(random_state=SEED, n_jobs=-1))])
    if name == "XGBoost":
        return ImbPipeline([("sm", smote()),
                            ("m", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                                random_state=SEED, n_jobs=-1))])
    raise ValueError(name)

SKLEARN_CLASSIFIERS = ["LASSO logistic", "PLS-DA", "SVM", "Random forest", "XGBoost"]

def fit_classifier(name, Xtr, ytr, cv_seed, grids):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cv_seed)
    return GridSearchCV(classifier_pipeline(name), grids[name], scoring="balanced_accuracy",
                        cv=cv, n_jobs=-1, refit=True).fit(Xtr, ytr)

def classifier_scores(name, search, X):
    """Continuous scores for AUC (PLS-DA uses its raw discriminant score)."""
    if name == "PLS-DA":
        return search.decision_function(X)
    return search.predict_proba(X)[:, 1]

# ---------- 1D-CNN classifier (same architecture and training as before) ----------
class CNNClassifier(nn.Module):
    def __init__(self, dropout=0.30):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(32, 16),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))
    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)

class CNNRegressor(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 32),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)

def train_net(net, X, y, lr, weight_decay, epochs, seed, loss):
    seed_everything(seed)
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(1),
                       torch.tensor(y, dtype=torch.float32))
    gen = torch.Generator(); gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=16, shuffle=True, generator=gen)
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            l = loss(net(xb), yb)
            l.backward(); opt.step()
    return net

def net_output(net, X):
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)).cpu().numpy()

CNN_CLS_GRID = list(ParameterGrid({"lr": [0.0005, 0.001], "weight_decay": [0.0001, 0.001],
                                   "dropout": [0.20, 0.40]}))

def run_cnn_classifier(Xtr, ytr, Xte, cv_seed):
    """Inner 3-fold CV (scaling + SMOTE inside folds), refit with 150 epochs."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=cv_seed)
    folds = list(cv.split(Xtr, ytr))
    tuning = []
    for prm in CNN_CLS_GRID:
        bas = []
        for f, (a, b) in enumerate(folds):
            sc = StandardScaler().fit(Xtr[a])
            Xa, ya = SMOTE(random_state=SEED + f, k_neighbors=3).fit_resample(sc.transform(Xtr[a]), ytr[a])
            net = train_net(CNNClassifier(prm["dropout"]), Xa, ya, prm["lr"], prm["weight_decay"],
                            150, SEED + f, nn.BCEWithLogitsLoss())
            prob = 1 / (1 + np.exp(-net_output(net, sc.transform(Xtr[b]))))
            bas.append(balanced_accuracy_score(ytr[b], (prob >= 0.5).astype(int)))
        tuning.append((np.mean(bas), prm))
    best = max(tuning, key=lambda t: t[0])[1]
    sc = StandardScaler().fit(Xtr)
    Xb, yb = smote().fit_resample(sc.transform(Xtr), ytr)
    net = train_net(CNNClassifier(best["dropout"]), Xb, yb, best["lr"], best["weight_decay"],
                    150, SEED, nn.BCEWithLogitsLoss())
    prob = 1 / (1 + np.exp(-net_output(net, sc.transform(Xte))))
    return (prob >= 0.5).astype(int), prob, best

# ---------- simple baselines (R1-3), fitted on training data only ----------
def _orient(f_tr, ytr):
    """+1 if higher feature values indicate the positive class on training data."""
    return 1.0 if roc_auc_score(ytr, f_tr) >= 0.5 else -1.0

def baseline_mean_absorbance(Xtr, ytr, Xte):
    ftr, fte = Xtr.mean(1), Xte.mean(1)
    sgn = _orient(ftr, ytr); ftr, fte = sgn * ftr, sgn * fte
    cands = np.unique(ftr)
    t = max(cands, key=lambda c: balanced_accuracy_score(ytr, (ftr >= c).astype(int)))
    return (fte >= t).astype(int), fte, {"threshold (mean absorbance)": float(sgn * t)}

def baseline_single_wavelength(Xtr, ytr, Xte):
    best = (-1.0, 0, 0.0, 1.0)
    for j in range(Xtr.shape[1]):
        f = Xtr[:, j]; sgn = _orient(f, ytr); f = sgn * f
        for t in np.quantile(f, np.linspace(0.02, 0.98, 49)):
            ba = balanced_accuracy_score(ytr, (f >= t).astype(int))
            if ba > best[0]:
                best = (ba, j, t, sgn)
    _, j, t, sgn = best
    fte = sgn * Xte[:, j]
    return (fte >= t).astype(int), fte, {"wavelength (nm)": float(wavelengths[j]),
                                         "threshold": float(sgn * t)}

# ======================================================================
# 5. REGRESSION MODELS
# ======================================================================
PLSR_GRID = {"m__n_components": [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]}
LASSO_GRID = {"m__alpha": np.logspace(-5, 1, 50)}
FUSED_FULL = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
FUSED_REPEAT = [0.001, 0.01, 0.05, 0.1]

def fit_linear(name, Xtr, ytr, cv_seed):
    cv = KFold(n_splits=3, shuffle=True, random_state=cv_seed)
    if name == "PLSR":
        pipe, grid = Pipeline([("s", StandardScaler()), ("m", PLSRegression(scale=False))]), PLSR_GRID
    else:
        pipe, grid = Pipeline([("s", StandardScaler()), ("m", Lasso(max_iter=100000))]), LASSO_GRID
    return GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=cv,
                        n_jobs=-1, refit=True).fit(Xtr, ytr)

def fused_lasso_fit(X, y, l1, l2):
    n, p = X.shape
    beta, b0 = cp.Variable(p), cp.Variable()
    obj = cp.sum_squares(y - b0 - X @ beta) / (2 * n) + l1 * cp.norm1(beta) + \
          l2 * cp.norm1(beta[1:] - beta[:-1])
    prob = cp.Problem(cp.Minimize(obj))
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)
    if beta.value is None:
        raise RuntimeError("Fused LASSO failed")
    return np.asarray(beta.value).ravel(), float(b0.value)

def run_fused(Xtr, ytr, Xte, cv_seed, grid):
    cv = KFold(n_splits=3, shuffle=True, random_state=cv_seed)
    folds = list(cv.split(Xtr))
    best = (np.inf, None, None)
    for l1 in grid:
        for l2 in grid:
            rm = []
            for a, b in folds:
                sc = StandardScaler().fit(Xtr[a])
                be, b0 = fused_lasso_fit(sc.transform(Xtr[a]), ytr[a], l1, l2)
                rm.append(np.sqrt(mean_squared_error(ytr[b], b0 + sc.transform(Xtr[b]) @ be)))
            if np.mean(rm) < best[0]:
                best = (np.mean(rm), l1, l2)
    _, l1, l2 = best
    sc = StandardScaler().fit(Xtr)
    be, b0 = fused_lasso_fit(sc.transform(Xtr), ytr, l1, l2)
    return b0 + sc.transform(Xte) @ be, {"lambda1": l1, "lambda2": l2}

CNN_REG_GRID = list(ParameterGrid({"lr": [0.0003, 0.001], "weight_decay": [0.0001, 0.001],
                                   "dropout": [0.20, 0.40]}))

def run_cnn_regressor(Xtr, ytr, Xte, cv_seed):
    """Inner 3-fold CV (150 epochs), refit on all training data (200 epochs, as published)."""
    cv = KFold(n_splits=3, shuffle=True, random_state=cv_seed)
    folds = list(cv.split(Xtr))
    tuning = []
    for prm in CNN_REG_GRID:
        rm = []
        for f, (a, b) in enumerate(folds):
            sc = StandardScaler().fit(Xtr[a])
            ym, ys = ytr[a].mean(), max(ytr[a].std(), 1e-12)
            net = train_net(CNNRegressor(prm["dropout"]), sc.transform(Xtr[a]), (ytr[a] - ym) / ys,
                            prm["lr"], prm["weight_decay"], 150, SEED + f, nn.MSELoss())
            pred = net_output(net, sc.transform(Xtr[b])) * ys + ym
            rm.append(np.sqrt(mean_squared_error(ytr[b], pred)))
        tuning.append((np.mean(rm), prm))
    best = min(tuning, key=lambda t: t[0])[1]
    sc = StandardScaler().fit(Xtr)
    ym, ys = ytr.mean(), max(ytr.std(), 1e-12)
    net = train_net(CNNRegressor(best["dropout"]), sc.transform(Xtr), (ytr - ym) / ys,
                    best["lr"], best["weight_decay"], 200, SEED, nn.MSELoss())
    return net_output(net, sc.transform(Xte)) * ys + ym, best

def clean_params(p):
    return "; ".join(f"{k.replace('m__', '')} = {v:.4g}" if isinstance(v, float) else
                     f"{k.replace('m__', '')} = {v}" for k, v in p.items())

# ======================================================================
# 6. DESCRIPTIVE STATISTICS AND OM/TC LABEL TABLE (S3 check, S5)
# ======================================================================
tr_c, te_c = train_test_split(idx, test_size=0.30, random_state=SEED, shuffle=True, stratify=y_OM)
tr_r, te_r = train_test_split(idx, test_size=0.30, random_state=SEED, shuffle=True)

rows = []
for var, v, yb in [("OM (%)", OM, y_OM), ("TC (%)", TC, y_TC)]:
    for sub, ii in [("Full dataset", idx), ("Training", tr_c), ("Test", te_c)]:
        a = v[ii]
        rows.append({"Variable": var, "Subset": sub, "n": len(a), "Mean": a.mean(),
                     "SD": a.std(ddof=1), "Min": a.min(), "Median": np.median(a),
                     "Max": a.max(), "n (above threshold)": int(yb[ii].sum())})
desc = pd.DataFrame(rows).round(2)
desc.to_csv(out("S3_descriptive_statistics_check.csv"), index=False)
print("\nDESCRIPTIVE STATISTICS (compare with Table S3)\n", desc.to_string(index=False))

both = int(((y_OM == 1) & (y_TC == 1)).sum())
om_only = int(((y_OM == 1) & (y_TC == 0)).sum())
tc_only = int(((y_OM == 0) & (y_TC == 1)).sum())
neither = int(((y_OM == 0) & (y_TC == 0)).sum())
agreement = (both + neither) / len(y_OM)
kappa = cohen_kappa_score(y_OM, y_TC)
r_omtc = np.corrcoef(OM, TC)[0, 1]
S5 = pd.DataFrame({"": ["OM > 4%", "OM ≤ 4%", "Total"],
                   "TC > 2.32%": [both, tc_only, both + tc_only],
                   "TC ≤ 2.32%": [om_only, neither, om_only + neither],
                   "Total": [both + om_only, tc_only + neither, len(y_OM)]})
S5_note = (f"Overall agreement = {agreement * 100:.1f}%; Cohen's κ = {kappa:.3f}; "
           f"Pearson r between continuous OM and TC = {r_omtc:.3f}.")
S5.to_csv(out("S5_OM_TC_label_table.csv"), index=False)
print("\nS5: OM vs TC LABELS\n", S5.to_string(index=False), "\n", S5_note)

# ======================================================================
# 7. ORIGINAL SPLIT — CLASSIFICATION (S4, S6, Figure 3)
# ======================================================================
print("\n=== Original split: classification ===")
orig_cls, hp_rows, roc_scores = [], [], {"OM": {}, "TC": {}}
for outcome, y in [("OM", y_OM), ("TC", y_TC)]:
    Xtr, Xte, ytr, yte = X_SG[tr_c], X_SG[te_c], y[tr_c], y[te_c]
    for name in SKLEARN_CLASSIFIERS:
        s = fit_classifier(name, Xtr, ytr, SEED, FULL_GRIDS)
        sc_ = classifier_scores(name, s, Xte)
        orig_cls.append({"Outcome": outcome, "Model": name, **cls_metrics(yte, s.predict(Xte), sc_)})
        hp_rows.append({"Task": f"{outcome} classification", "Model": name,
                        "Selected hyperparameters": clean_params(s.best_params_)})
        roc_scores[outcome][name] = sc_
        print(f"  {outcome} {name} done")
    pred, prob, best = run_cnn_classifier(Xtr, ytr, Xte, SEED)
    orig_cls.append({"Outcome": outcome, "Model": "1D-CNN", **cls_metrics(yte, pred, prob)})
    hp_rows.append({"Task": f"{outcome} classification", "Model": "1D-CNN",
                    "Selected hyperparameters": clean_params(best)})
    roc_scores[outcome]["1D-CNN"] = prob
    print(f"  {outcome} 1D-CNN done")
    for bname, bfun in [("Baseline: mean absorbance", baseline_mean_absorbance),
                        ("Baseline: single wavelength", baseline_single_wavelength)]:
        p, f, info = bfun(Xtr, ytr, Xte)
        orig_cls.append({"Outcome": outcome, "Model": bname, **cls_metrics(yte, p, f)})
        hp_rows.append({"Task": f"{outcome} classification", "Model": bname,
                        "Selected hyperparameters": clean_params(info)})
orig_cls = pd.DataFrame(orig_cls)
orig_cls.round(4).to_csv(out("S6a_original_split_classification.csv"), index=False)
print(orig_cls.round(3).to_string(index=False))

# Figure 3: ROC curves, panel (a) OM and (b) TC
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, (lab, outcome, yte) in zip(axes, [("(a)", "OM", y_OM[te_c]), ("(b)", "TC", y_TC[te_c])]):
    for name, sc_ in roc_scores[outcome].items():
        fpr, tpr, _ = roc_curve(yte, sc_)
        ax.step(fpr, tpr, where="post", lw=1.8, label=f"{name} (AUC = {roc_auc_score(yte, sc_):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    thr = "OM > 4%" if outcome == "OM" else "TC > 2.32%"
    ax.set_title(f"{lab} {outcome}-based classes ({thr})", loc="left")
    ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(out("Figure_3_ROC_OM_TC.png"), dpi=600, bbox_inches="tight")
plt.close()

# ======================================================================
# 8. ORIGINAL SPLIT — REGRESSION (S4, S6) AND FIGURE 4
# ======================================================================
print("\n=== Original split: regression ===")
orig_reg, lin_models = [], {}
for outcome, yc in [("OM", OM), ("TC", TC)]:
    Xtr, Xte, ytr, yte = X_SG[tr_r], X_SG[te_r], yc[tr_r], yc[te_r]
    for name in ["PLSR", "LASSO"]:
        s = fit_linear(name, Xtr, ytr, SEED)
        lin_models[(outcome, name)] = s.best_estimator_
        orig_reg.append({"Outcome": outcome, "Model": name, **reg_metrics(yte, s.predict(Xte))})
        hp_rows.append({"Task": f"{outcome} regression", "Model": name,
                        "Selected hyperparameters": clean_params(s.best_params_)})
    p, info = run_fused(Xtr, ytr, Xte, SEED, FUSED_FULL)
    orig_reg.append({"Outcome": outcome, "Model": "Fused LASSO", **reg_metrics(yte, p)})
    hp_rows.append({"Task": f"{outcome} regression", "Model": "Fused LASSO",
                    "Selected hyperparameters": clean_params(info)})
    p, best = run_cnn_regressor(Xtr, ytr, Xte, SEED)
    orig_reg.append({"Outcome": outcome, "Model": "1D-CNN", **reg_metrics(yte, p)})
    hp_rows.append({"Task": f"{outcome} regression", "Model": "1D-CNN",
                    "Selected hyperparameters": clean_params(best)})
    print(f"  {outcome} regression done")
orig_reg = pd.DataFrame(orig_reg)
orig_reg.round(4).to_csv(out("S6b_original_split_regression.csv"), index=False)
print(orig_reg.round(3).to_string(index=False))

S4 = pd.DataFrame(hp_rows)
S4.to_csv(out("S4_selected_hyperparameters.csv"), index=False)
S4_note = ("Repeated-split evaluation used the same grids except: Random forest — n_estimators = 300, "
           "max_depth ∈ {None, 5}, min_samples_leaf ∈ {1, 4}, max_features = sqrt; XGBoost — "
           "n_estimators = 300, learning_rate ∈ {0.03, 0.10}, max_depth ∈ {2, 3}, min_child_weight ∈ {1, 5}, "
           "subsample = 0.8, colsample_bytree = 0.5, reg_alpha ∈ {0, 1}, reg_lambda ∈ {1, 10}; fused LASSO — "
           "λ1, λ2 ∈ {0.001, 0.01, 0.05, 0.1}.")

def vip_scores(pls):
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    ss = np.sum(t ** 2, axis=0) * q.ravel() ** 2
    wn = (w / np.linalg.norm(w, axis=0)) ** 2
    return np.sqrt(w.shape[0] * (wn @ ss) / ss.sum())

def contiguous_ranges(wl):
    if len(wl) == 0:
        return "none"
    step = np.median(np.diff(wavelengths))
    groups, start, prev = [], wl[0], wl[0]
    for w in wl[1:]:
        if w - prev > 1.5 * step:
            groups.append((start, prev)); start = w
        prev = w
    groups.append((start, prev))
    return ", ".join(f"{a:.0f}–{b:.0f}" if b > a else f"{a:.0f}" for a, b in groups)

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
interp = {}
for k, outcome in enumerate(["OM", "TC"]):
    coef = lin_models[(outcome, "LASSO")].named_steps["m"].coef_
    vip = vip_scores(lin_models[(outcome, "PLSR")].named_steps["m"])
    axes[0, k].bar(wavelengths, coef, width=np.median(np.diff(wavelengths)), color="#1f77b4")
    axes[0, k].axhline(0, color="black", lw=0.6)
    axes[0, k].set_title(f"({'ab'[k]}) LASSO coefficients — {outcome}", loc="left")
    axes[1, k].plot(wavelengths, vip, color="#d62728")
    axes[1, k].axhline(1, ls="--", color="grey")
    axes[1, k].set_title(f"({'cd'[k]}) PLSR VIP scores — {outcome}", loc="left")
    axes[1, k].set_xlabel("Wavelength (nm)")
    for ax in axes[:, k]:
        for lo, hi in [(1100, 1200), (1380, 1440)]:
            ax.axvspan(lo, hi, color="orange", alpha=0.18)
    interp[outcome] = {"n_selected": int(np.sum(coef != 0)),
                       "selected": contiguous_ranges(wavelengths[coef != 0]),
                       "vip_regions": contiguous_ranges(wavelengths[vip > 1])}
axes[0, 0].set_ylabel("Standardized coefficient")
axes[1, 0].set_ylabel("VIP")
plt.tight_layout()
plt.savefig(out("Figure_4_LASSO_coefficients_PLS_VIP.png"), dpi=600, bbox_inches="tight")
plt.close()
pd.DataFrame(interp).T.to_csv(out("Figure4_selected_wavelengths.csv"))

# ======================================================================
# 9. REPEATED SPLITS — CLASSIFICATION (S7)
# ======================================================================
def resume(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def done_set(d, keys):
    return set(map(tuple, d[keys].values.tolist())) if len(d) else set()

path = out("S7_raw_repeated_classification.csv")
rep_cls = resume(path)
done = done_set(rep_cls, ["Repeat", "Outcome", "Model"])
T0 = time.time()
print("\n=== Repeated splits: classification ===")
for r in range(N_REPEATS):
    new = []
    for outcome, y in [("OM", y_OM), ("TC", y_TC)]:
        tr, te = train_test_split(idx, test_size=0.30, random_state=1000 + r, stratify=y)
        Xtr, Xte, ytr, yte = X_SG[tr], X_SG[te], y[tr], y[te]
        for name in SKLEARN_CLASSIFIERS:
            if (r, outcome, name) in done:
                continue
            s = fit_classifier(name, Xtr, ytr, 1000 + r, REPEAT_GRIDS)
            new.append({"Repeat": r, "Outcome": outcome, "Model": name,
                        **cls_metrics(yte, s.predict(Xte), classifier_scores(name, s, Xte))})
        for bname, bfun in [("Baseline: mean absorbance", baseline_mean_absorbance),
                            ("Baseline: single wavelength", baseline_single_wavelength)]:
            if (r, outcome, bname) in done:
                continue
            p, f, _ = bfun(Xtr, ytr, Xte)
            new.append({"Repeat": r, "Outcome": outcome, "Model": bname, **cls_metrics(yte, p, f)})
        if r < N_SLOW and (r, outcome, "1D-CNN") not in done:
            p, prob, _ = run_cnn_classifier(Xtr, ytr, Xte, 1000 + r)
            new.append({"Repeat": r, "Outcome": outcome, "Model": "1D-CNN", **cls_metrics(yte, p, prob)})
    if new:
        rep_cls = pd.concat([rep_cls, pd.DataFrame(new)], ignore_index=True)
        rep_cls.to_csv(path, index=False)
    print(f"  classification repeat {r + 1}/{N_REPEATS}  ({(time.time() - T0) / 60:.1f} min elapsed)", flush=True)

cls_order = SKLEARN_CLASSIFIERS + ["1D-CNN", "Baseline: mean absorbance", "Baseline: single wavelength"]
rep_cls["Model"] = pd.Categorical(rep_cls["Model"], cls_order, ordered=True)
rep_cls = rep_cls.sort_values(["Outcome", "Model", "Repeat"])
cls_metric_cols = ["Balanced accuracy", "Sensitivity", "Specificity", "Precision (PPV)", "AUC"]
S7 = fmt_summary(rep_cls, ["Outcome", "Model"], cls_metric_cols)
S7.to_csv(out("S7_repeated_classification_summary.csv"), index=False)
print(S7.to_string(index=False))

# ======================================================================
# 10. REPEATED SPLITS — REGRESSION WITH PREPROCESSING (S8)
# ======================================================================
path = out("S8_raw_repeated_regression.csv")
rep_reg = resume(path)
done = done_set(rep_reg, ["Repeat", "Outcome", "Preprocessing", "Model"])
T0 = time.time()
print("\n=== Repeated splits: regression ===")
for r in range(N_REPEATS):
    new = []
    tr, te = train_test_split(idx, test_size=0.30, random_state=2000 + r)
    for outcome, yc in [("OM", OM), ("TC", TC)]:
        for prep in PREPROCESSING:
            Xtr, Xte = preprocess(prep, tr, te)
            for name in ["PLSR", "LASSO"]:
                if (r, outcome, prep, name) in done:
                    continue
                s = fit_linear(name, Xtr, yc[tr], 2000 + r)
                new.append({"Repeat": r, "Outcome": outcome, "Preprocessing": prep, "Model": name,
                            **reg_metrics(yc[te], s.predict(Xte))})
        if r < N_SLOW:
            Xtr, Xte = preprocess("SG", tr, te)
            if (r, outcome, "SG", "Fused LASSO") not in done:
                p, _ = run_fused(Xtr, yc[tr], Xte, 2000 + r, FUSED_REPEAT)
                new.append({"Repeat": r, "Outcome": outcome, "Preprocessing": "SG",
                            "Model": "Fused LASSO", **reg_metrics(yc[te], p)})
            if (r, outcome, "SG", "1D-CNN") not in done:
                p, _ = run_cnn_regressor(Xtr, yc[tr], Xte, 2000 + r)
                new.append({"Repeat": r, "Outcome": outcome, "Preprocessing": "SG",
                            "Model": "1D-CNN", **reg_metrics(yc[te], p)})
    if new:
        rep_reg = pd.concat([rep_reg, pd.DataFrame(new)], ignore_index=True)
        rep_reg.to_csv(path, index=False)
    print(f"  regression repeat {r + 1}/{N_REPEATS}  ({(time.time() - T0) / 60:.1f} min elapsed)", flush=True)

rep_reg["Preprocessing"] = pd.Categorical(rep_reg["Preprocessing"], PREPROCESSING, ordered=True)
rep_reg["Model"] = pd.Categorical(rep_reg["Model"], ["PLSR", "LASSO", "Fused LASSO", "1D-CNN"], ordered=True)
rep_reg = rep_reg.sort_values(["Outcome", "Model", "Preprocessing", "Repeat"])
rep_reg = rep_reg.dropna(subset=["Model", "Preprocessing"])
reg_metric_cols = ["R2", "RMSE", "Bias", "RPD", "RPIQ"]
S8 = fmt_summary(rep_reg, ["Outcome", "Model", "Preprocessing"], reg_metric_cols)
S8 = S8[S8["Repeats"] > 0]
S8.to_csv(out("S8_repeated_regression_summary.csv"), index=False)
print(S8.to_string(index=False))

# ======================================================================
# 11. SUPPLEMENTARY TABLES S4–S8 AS A WORD FILE
# ======================================================================
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.section import WD_ORIENT

def add_table(doc, title, data, note=None, digits=3):
    p = doc.add_paragraph(); run = p.add_run(title); run.bold = True
    data = data.copy()
    for c in data.columns:
        if pd.api.types.is_float_dtype(data[c]):
            data[c] = data[c].map(lambda v: f"{v:.{digits}f}")
    t = doc.add_table(rows=1, cols=len(data.columns)); t.style = "Table Grid"
    for i, c in enumerate(data.columns):
        cell = t.rows[0].cells[i]; cell.text = str(c)
        cell.paragraphs[0].runs[0].bold = True
    for _, row in data.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    for row in t.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                for rr in par.runs:
                    rr.font.size = Pt(8)
    if note:
        n = doc.add_paragraph(); nr = n.add_run("Note. " + note); nr.font.size = Pt(9)
    doc.add_paragraph()

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
sec = doc.sections[0]
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = sec.page_height, sec.page_width
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(1.8))

add_table(doc, "Table S4. Selected hyperparameters of all models (Savitzky–Golay-smoothed spectra, original split)",
          S4, S4_note)
add_table(doc, f"Table S5. Cross-tabulation of OM-based (OM > 4%) and TC-based (TC > 2.32%) class labels (n = {len(y_OM)})",
          S5, S5_note)
s6a = orig_cls[["Outcome", "Model", "TP", "FP", "FN", "TN", "Sensitivity", "Specificity",
                "Precision (PPV)", "Balanced accuracy", "AUC"]]
add_table(doc, "Table S6a. Confusion matrices and precision on the original 30% test set (classification split)",
          s6a, "TP, FP, FN, TN = true positives, false positives, false negatives, true negatives; "
               "precision (PPV) = TP/(TP + FP).")
add_table(doc, "Table S6b. Regression performance on the original 30% test set (regression split)",
          orig_reg[["Outcome", "Model", "R2", "RMSE", "Bias", "RPD", "RPIQ"]],
          "Bias = mean(predicted − observed); RPD = SD/RMSE; RPIQ = IQR/RMSE.")
add_table(doc, "Table S7. Classification performance over repeated stratified 70/30 splits", S7,
          f"Mean ± SD [2.5th–97.5th percentile]; {N_REPEATS} repeats"
          + ("." if N_SLOW == N_REPEATS else f" ({N_SLOW} for the 1D-CNN)."))
add_table(doc, "Table S8. Regression performance over repeated random 70/30 splits by preprocessing", S8,
          f"Mean ± SD [2.5th–97.5th percentile]; {N_REPEATS} repeats"
          + (" for all models; fused LASSO and the 1D-CNN with SG spectra only." if N_SLOW == N_REPEATS
             else f" for PLSR and LASSO, {N_SLOW} for fused LASSO and the 1D-CNN (SG spectra only)."))
doc.save(out("Supplementary_Tables_S4_S8.docx"))

# ======================================================================
# 12. VALUES FOR THE MANUSCRIPT PLACEHOLDERS
# ======================================================================
def ms(d_, col):
    return f"{d_[col].mean():.3f} ± {d_[col].std():.3f}"

L = []
add = L.append
multi = [m for m in SKLEARN_CLASSIFIERS + ["1D-CNN"]]
add("=== VALUES FOR THE [XX] PLACEHOLDERS IN THE MANUSCRIPT ===\n")
add(f"Methods (labels, Table S5): OM-positive samples also TC-positive = {both} of {both + om_only}; "
    f"TC-positive only = {tc_only}; agreement = {agreement * 100:.1f}%; Cohen's kappa = {kappa:.3f}")
for outcome in ["OM", "TC"]:
    d_ = rep_cls[rep_cls.Outcome == outcome]
    means = d_[d_.Model.isin(multi)].groupby("Model", observed=True)["Balanced accuracy"].agg(["mean", "std"])
    best_m, worst_m = means["mean"].idxmax(), means["mean"].idxmin()
    ppv = d_[d_.Model.isin(multi)].groupby("Model", observed=True)["Precision (PPV)"].mean()
    base_mean = d_[d_.Model == "Baseline: mean absorbance"]
    base_wl = d_[d_.Model == "Baseline: single wavelength"]
    add(f"\n[{outcome}] classification, repeated splits:")
    add(f"  best classifier: {best_m}, balanced accuracy {means.loc[best_m, 'mean']:.3f} ± {means.loc[best_m, 'std']:.3f}")
    add(f"  lowest classifier: {worst_m}, balanced accuracy {means.loc[worst_m, 'mean']:.3f}")
    add(f"  between-split SD range of balanced accuracy: {means['std'].min():.3f}–{means['std'].max():.3f}")
    add(f"  baseline mean absorbance: balanced accuracy {ms(base_mean, 'Balanced accuracy')}")
    add(f"  baseline single wavelength: balanced accuracy {ms(base_wl, 'Balanced accuracy')}")
    add(f"  mean precision range of classifiers: {ppv.min():.3f}–{ppv.max():.3f}")
for outcome in ["OM", "TC"]:
    o = orig_reg[orig_reg.Outcome == outcome]
    d_ = rep_reg[rep_reg.Outcome == outcome]
    add(f"\n[{outcome}] regression:")
    add(f"  original split: RPD {o.RPD.min():.2f}–{o.RPD.max():.2f}; RPIQ {o.RPIQ.min():.2f}–{o.RPIQ.max():.2f}; "
        f"bias {o.Bias.min():.3f} to {o.Bias.max():.3f}")
    for name in ["LASSO", "PLSR"]:
        add(f"  repeated SG {name}: R2 {ms(d_[(d_.Model == name) & (d_.Preprocessing == 'SG')], 'R2')}")
    pl = d_[d_.Model == "PLSR"].groupby("Preprocessing", observed=True)["R2"].mean()
    add(f"  PLSR mean R2 by preprocessing: " + "; ".join(f"{k} {v:.3f}" for k, v in pl.items()))
    add(f"  best PLSR preprocessing: {pl.idxmax()} (mean R2 {pl.max():.3f}; published PLSR: "
        f"{'0.79' if outcome == 'OM' else '0.78'})")
    add(f"  repeated 1D-CNN (SG): R2 {ms(d_[d_.Model == '1D-CNN'], 'R2')}")
for outcome in ["OM", "TC"]:
    add(f"\n[{outcome}] Figure 4: LASSO kept {interp[outcome]['n_selected']} of {len(wavelengths)} wavelengths "
        f"({interp[outcome]['selected']}); PLSR VIP > 1 in: {interp[outcome]['vip_regions']} nm")
text = "\n".join(L)
open(out("numbers_for_manuscript.txt"), "w").write(text)
print("\n" + text)
print(f"\nAll outputs saved in the folder '{OUT}'.")
