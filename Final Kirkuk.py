#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ======================================================================
# ATTEN-SPEC-TOMO (A.S.T.) v4.0 — KIRKUK IRAQ REGION
# written bu Dr. Shaheen MOhammed Saleh Ahmed 
# ======================================================================
# Changelog from v3.1:
#   ── BUG FIXES ──
#   1. run_inversion: gradient-chain break on non-reg iterations fixed
#      (data_only was a Python float fed to torch.tensor, detaching the
#       computational graph; now we simply skip the costly reg terms
#       while keeping the data term differentiable)
#   2. _region_stats: additional guard against nz==0, dz<=0 edge cases
#   3. coda_decay_proxy: safe handling when Hilbert transform yields NaN
#   4. compute_path_features: guarded against zero-length FFT segments
#   5. band_snr_db: clamped output to [-100, 100] to avoid inf
#   6. build_L_sparse: added explicit sort for CSR correctness on GPU
#   7. _save_geotiff_slices: flip array for north-up orientation
#   8. pick_onset: guarded against all-zero filtered traces
#   9. _apply_band: added NaN/Inf check before filtering
#  10. haversine_km: guarded against identical points (dist=0)
#
#   ── VISUALIZATION UPGRADES ──
#   • DPI raised to 400 (savefig 480) for publication-quality output
#   • Custom high-contrast scientific colormaps per field type
#   • Contour overlays on all map slices
#   • Geographic grid lines with degree labels
#   • Enhanced typography (DejaVu Sans, heavier weights)
#   • Depth cross-section plots (lon–depth and lat–depth)
#   • Improved residual diagnostics with KDE overlay
#   • Coverage vs. uncertainty with marginal histograms
#   • Frequency-dependent Q curves with shaded uncertainty bands
#   • Subregion bar charts with error bars
#   • Consistent tick formatting and axis labels
#   • All colorbars use ScalarFormatter for readability
# ======================================================================

from __future__ import annotations
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import os, glob, math, json, csv, random, warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import scipy.signal as sps
from scipy.signal import resample_poly, sosfiltfilt, butter, decimate
from scipy.fft import rfft as sp_rfft, rfftfreq as sp_rfftfreq, next_fast_len
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, ScalarFormatter, AutoMinorLocator
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════════
#  CUSTOM COLORMAPS — high contrast, perceptually balanced
# ══════════════════════════════════════════════════════════════════════
def _build_cmap(colors, name, N=512):
    return LinearSegmentedColormap.from_list(name, colors, N=N)

# Intrinsic Q — deep teal → electric cyan → warm gold → hot magenta
CMAP_QI = _build_cmap([
    "#0a0a2e", "#0b3d6b", "#0891b2", "#06d6a0",
    "#fbbf24", "#f97316", "#ef4444", "#db2777"
], "ast_qi")

# Scattering Q — midnight → royal purple → neon pink → solar orange
CMAP_QSCT = _build_cmap([
    "#0f0720", "#3b0764", "#7c3aed", "#c026d3",
    "#f43f5e", "#fb923c", "#fde047", "#fefce8"
], "ast_qsct")

# Uncertainty — black → indigo → crimson → bright yellow
CMAP_STD = _build_cmap([
    "#000000", "#1e1b4b", "#4c1d95", "#9333ea",
    "#dc2626", "#f97316", "#facc15", "#fefce8"
], "ast_std")

# Coverage — deep ocean → teal → lime → white-hot
CMAP_COV = _build_cmap([
    "#000814", "#001d3d", "#003566", "#0077b6",
    "#00b4d8", "#43aa8b", "#90be6d", "#f9c74f", "#f8961e"
], "ast_cov")

# Residuals — diverging blue ↔ red
CMAP_RESID = _build_cmap([
    "#08306b", "#2171b5", "#6baed6", "#c6dbef",
    "#ffffff",
    "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"
], "ast_resid")

# Histogram — rich gradient
CMAP_HIST = _build_cmap([
    "#14213d", "#3a0ca3", "#7209b7", "#f72585"
], "ast_hist")

# Scatter — vivid categorical-safe
CMAP_SCAT = _build_cmap([
    "#1b4332", "#2d6a4f", "#40916c", "#74c69d",
    "#ffd166", "#ef476f", "#9b2226"
], "ast_scat")

# ──────────────────────────────────────────────────────────────────────
#  GLOBAL PLOTTING STYLE — publication-quality, high-res
# ──────────────────────────────────────────────────────────────────────
_PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
    "#f4a261", "#264653", "#a855f7", "#06b6d4"
]

mpl.rcParams.update({
    # ── Resolution ──
    "figure.dpi":           400,
    "savefig.dpi":          480,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "#fafafa",
    "savefig.pad_inches":   0.15,

    # ── Typography ──
    "font.family":          "sans-serif",
    "font.sans-serif":      ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":            13,
    "font.weight":          "medium",

    # ── Axes ──
    "axes.titlesize":       17,
    "axes.titleweight":     "bold",
    "axes.titlepad":        14,
    "axes.labelsize":       14.5,
    "axes.labelweight":     "bold",
    "axes.labelpad":        8,
    "axes.linewidth":       1.3,
    "axes.edgecolor":       "#2f2f2f",
    "axes.facecolor":       "#fafafa",
    "axes.grid":            False,
    "axes.prop_cycle":      matplotlib.cycler(color=_PALETTE),

    # ── Ticks ──
    "xtick.labelsize":      12,
    "ytick.labelsize":      12,
    "xtick.major.width":    1.1,
    "ytick.major.width":    1.1,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "xtick.direction":      "in",
    "ytick.direction":      "in",

    # ── Grid ──
    "grid.linestyle":       ":",
    "grid.alpha":           0.35,
    "grid.linewidth":       0.8,
    "grid.color":           "#888888",

    # ── Lines ──
    "lines.linewidth":      2.4,
    "lines.markersize":     7,

    # ── Legend ──
    "legend.fontsize":      11.5,
    "legend.frameon":       True,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#444444",
    "legend.fancybox":      True,
    "legend.shadow":        True,
    "legend.borderpad":     0.6,

    # ── Figure ──
    "figure.facecolor":     "#fafafa",
    "figure.autolayout":    False,
})


# ── Seismology ──
from obspy import read
from obspy.core.trace import Trace
try:
    from obspy.io.sac import SACTrace
    _HAS_SACTRACE = True
except Exception:
    _HAS_SACTRACE = False

# ── Geospatial ──
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from pyproj import Geod

# ── Progress ──
from tqdm import tqdm


def _lazy_import_torch_stack():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


# ╔══════════════════════════════════════════════════════════════════════╗
# ║   CONFIGURATION — KIRKUK / IRAQ REGION                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
@dataclass
class Config:
    sac_root: str = r"C:\Users\shahe\Desktop\Kirkuk SAC files\Waveforms"
    out_dir:  str = r"./ast_outputs_kirkuk_v4"

    # Region bounds
    min_lon: float = 38.0;   max_lon: float = 49.0
    min_lat: float = 29.0;   max_lat: float = 39.0
    min_depth_km: float = 0.0;  max_depth_km: float = 15.0

    # Grid
    nx: int = 100;  ny: int = 108;  nz: int = 16

    # Frequency bands
    bands: List[Tuple[float, float]] = (
        (0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0)
    )

    # Windows
    pre_pick:  float = 0.05
    post_pick: float = 2.0
    coda_lapses: List[Tuple[float, float]] = ((30.0, 40.0), (40.0, 55.0))

    # Spectral / attenuation
    geometric_gamma_init: float = 1.0
    min_snr_db: float = -5.0
    target_fs:  float = 80.0

    # Fast mode
    fast_mode: bool = True
    short_post_pick: float = 1.0

    # Regularization
    lambda_tv:    float = 0.03
    lambda_aniso: float = 0.03

    # Learned prior
    lambda_pnp:    float = 0.03
    pnp_noise_std: float = 0.015
    pnp_dropout_p: float = 0.10

    max_iter_inv: int = 300
    lr_inv: float = 3e-2

    # Ensemble
    n_ensemble:      int   = 10
    bootstrap_frac:  float = 0.85
    mc_prior_samples: int  = 4

    # Faults
    faults_geojson: Optional[str] = None

    # Subregions
    subregions: Dict[str, List[Tuple[float, float]]] = None

    # Reproducibility
    seed: int = 123
    debug_feature_extraction: bool = True

    # ── NEW v4: visualization controls ──
    contour_levels: int = 10
    map_figsize: Tuple[float, float] = (11.0, 7.0)
    cross_section_figsize: Tuple[float, float] = (12.0, 5.5)


cfg = Config()

# ──────────────────────────────────────────────────────────────────────
# Subregion defaults (depth ranges clamped to max_depth_km=15)
# ──────────────────────────────────────────────────────────────────────
if cfg.subregions is None:
    cfg.subregions = {
        "Kirkuk_Embayment_0_15km": [
            (43.8, 35.0), (45.0, 35.0), (45.0, 36.0), (43.8, 36.0)
        ],
        "Zagros_Foothills_0_15km": [
            (45.0, 35.0), (46.5, 35.0), (46.5, 36.5), (45.0, 36.5)
        ],
        "Mesopotamian_Foredeep_0_15km": [
            (43.0, 31.5), (45.5, 31.5), (45.5, 34.0), (43.0, 34.0)
        ],
        "Hamrin_Makhul_0_15km": [
            (43.0, 34.0), (44.5, 34.0), (44.5, 35.5), (43.0, 35.5)
        ],
        "Kirkuk_Deep_Crust_10_15km": [
            (43.5, 34.5), (45.5, 34.5), (45.5, 36.5), (43.5, 36.5)
        ],
        "NorthIraq_SETurkey_0_15km": [
            (42.5, 36.5), (44.5, 36.5), (44.5, 38.0), (42.5, 38.0)
        ],
    }

os.makedirs(cfg.out_dir, exist_ok=True)
with open(os.path.join(cfg.out_dir, 'config.json'), 'w', encoding="utf-8") as f:
    json.dump(asdict(cfg), f, indent=2)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
#  NON-TORCH UTILITIES
# ══════════════════════════════════════════════════════════════════════
geod = Geod(ellps='WGS84')


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance; returns 0.0 for identical points."""
    if abs(lon1 - lon2) < 1e-10 and abs(lat1 - lat2) < 1e-10:
        return 0.0
    try:
        _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
        return max(0.0, dist_m / 1000.0)
    except Exception:
        return 0.0


def _safe_norm_band(fs: float, f1: float, f2: float) -> Optional[Tuple[float, float]]:
    if fs is None or fs <= 0:
        return None
    nyq = 0.5 * fs
    low  = max(1e-3, min(abs(f1), abs(f2)))
    high = max(1e-2, max(abs(f1), abs(f2)))
    low  = min(low,  0.95 * nyq)
    high = min(high, 0.95 * nyq)
    if low >= high:
        low, high = 0.05 * nyq, 0.45 * nyq
    lo_n = low / nyq
    hi_n = high / nyq
    if not (0.0 < lo_n < hi_n < 1.0):
        return None
    return lo_n, hi_n


@lru_cache(maxsize=256)
def _sos_for_band(fs: float, f1: float, f2: float, order: int = 3):
    bands = _safe_norm_band(fs, f1, f2)
    if bands is None:
        return None
    lo, hi = bands
    return butter(order, [lo, hi], btype='band', output='sos')


def _apply_band(x: np.ndarray, fs: float, f1: float, f2: float):
    """Band-pass filter with NaN/Inf guard."""
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    sos = _sos_for_band(fs, f1, f2)
    if sos is None:
        return x
    try:
        return sosfiltfilt(sos, x)
    except Exception:
        return x


def _safe_butter_band(fs: float, f1: float, f2: float, order: int = 3):
    bands = _safe_norm_band(fs, f1, f2)
    if bands is None:
        return None, None
    lo_n, hi_n = bands
    b, a = sps.butter(order, [lo_n, hi_n], btype='band')
    return b, a


# ──────────────────────────────────────────────────────────────────────
#  FAST SAC LOADER
# ──────────────────────────────────────────────────────────────────────
def fast_read_sac(fp):
    try:
        if _HAS_SACTRACE:
            st = SACTrace.read(fp)
            x = np.asarray(st.data, dtype=np.float32, order="C")
            fs = float(1.0 / st.delta) if getattr(st, "delta", 0.0) else None
            return x, fs, st
        else:
            tr = read(fp)[0]
            return (tr.data.astype(np.float32, copy=False),
                    float(tr.stats.sampling_rate), tr.stats.sac)
    except Exception:
        return None, None, None


# ══════════════════════════════════════════════════════════════════════
#  DATA DISCOVERY & METADATA
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PathFeature:
    event_id: str;  station: str
    lon_e: float;   lat_e: float;   dep_e_km: float
    lon_s: float;   lat_s: float;   elev_s_km: float


class SACCatalogue:
    def __init__(self, root: str):
        self.root = root
        self.files = sorted(
            glob.glob(os.path.join(root, '**', '*.SAC'), recursive=True)
        )
        if not self.files:
            self.files = sorted(
                glob.glob(os.path.join(root, '**', '*.sac'), recursive=True)
            )
        print(f"Discovered {len(self.files)} SAC files")

    def read_meta(self, fp: str):
        tr = read(fp, headonly=True)[0]
        h = tr.stats.sac if hasattr(tr.stats, 'sac') else None
        meta = {}
        if h:
            meta = dict(
                kstnm=str(getattr(h, 'kstnm', 'STA')),
                kevnm=str(getattr(h, 'kevnm', 'EV')),
                stla=float(getattr(h, 'stla', np.nan)),
                stlo=float(getattr(h, 'stlo', np.nan)),
                stel=float(getattr(h, 'stel', 0.0)),
                evla=float(getattr(h, 'evla', np.nan)),
                evlo=float(getattr(h, 'evlo', np.nan)),
                evdp=float(getattr(h, 'evdp', np.nan)),
            )
        return meta

    def build_paths(self) -> List[PathFeature]:
        feats: List[PathFeature] = []
        for fp in tqdm(self.files, desc='Reading headers'):
            try:
                m = self.read_meta(fp)
                if any(np.isnan([m['stla'], m['stlo'], m['evla'], m['evlo']])):
                    continue
                dep_raw = float(m['evdp'])
                dep_km = dep_raw / 1000.0 if dep_raw > 100 else dep_raw
                pf = PathFeature(
                    event_id=m['kevnm'], station=m['kstnm'],
                    lon_e=m['evlo'], lat_e=m['evla'],
                    dep_e_km=dep_km,
                    lon_s=m['stlo'], lat_s=m['stla'],
                    elev_s_km=float(m['stel']) / 1000.0,
                )
                feats.append(pf)
            except Exception:
                pass
        print(f"Usable paths: {len(feats)}")
        return feats


cat = SACCatalogue(cfg.sac_root)
PATHS = cat.build_paths()

FILE_INDEX: Dict[Tuple[str, str], List[str]] = {}
for fp in cat.files:
    try:
        h = read(fp, headonly=True)[0].stats.sac
        key = (str(getattr(h, 'kevnm', 'EV')),
               str(getattr(h, 'kstnm', 'STA')))
        FILE_INDEX.setdefault(key, []).append(fp)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════
#  PICKING & WINDOWING
# ══════════════════════════════════════════════════════════════════════
def pick_onset(tr: Trace, p_or_s: str = 'S') -> Optional[float]:
    try:
        data = tr.data.astype(float)
        fs = tr.stats.sampling_rate
        if len(data) < 100:
            return len(data) / fs * 0.5
        x = _apply_band(data, fs, 1.0, 15.0)
        # Guard against all-zero filtered trace
        if np.max(np.abs(x)) < 1e-20:
            return len(data) / fs * 0.3
        nsta = max(1, int(0.5 * fs))
        nlta = max(2, int(5 * fs))
        if len(x) < nlta + 10:
            return max(0.0, min(len(x) / fs, 1.0))
        sta_sum = sps.lfilter(np.ones(nsta) / max(1, nsta), 1, x ** 2)
        lta_sum = sps.lfilter(np.ones(nlta) / max(1, nlta), 1, x ** 2)
        cft = sta_sum / (lta_sum + 1e-9)
        i = int(np.argmax(cft[nlta:])) + nlta
        return i / fs
    except Exception:
        return None


def make_windows(pick_s, fs, pre, post, coda_lapses):
    if pick_s is None:
        pick_s = 1.0
    dw = (max(0.0, pick_s - pre), pick_s + post)
    codas = [(pick_s + a, pick_s + b) for (a, b) in coda_lapses]
    return dw, codas


def band_snr_db(x, fs, f1, f2):
    try:
        y = _apply_band(x, fs, f1, f2)
        s = float(np.percentile(np.abs(y), 95))
        n_ = float(np.median(np.abs(y)) + 1e-9)
        val = 20 * np.log10((s + 1e-6) / (n_ + 1e-6))
        return float(np.clip(val, -100.0, 100.0))
    except Exception:
        return -10.0


# ──────────────────────────────────────────────────────────────────────
#  CHANNEL PICKING
# ──────────────────────────────────────────────────────────────────────
PREFERRED_CHANNELS = (
    "HHE", "BHE", "ENE", "EHE", "EH2", "HH2",
    "HHN", "BHN", "ENN", "EHN", "EH1", "HH1",
    "HHZ", "BHZ", "HH", "BH", "EH", "SH", "LH",
)


def _pick_preferred_file(fp_list):
    for ch in PREFERRED_CHANNELS:
        for fp in fp_list:
            name = os.path.basename(fp).upper()
            if ch in name:
                return fp
    return fp_list[0] if fp_list else None


# ══════════════════════════════════════════════════════════════════════
#  SPECTRAL FEATURES
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PathBandFeatures:
    tstar_dir: float;    tstar_coda_i: float;  tstar_coda_s: float
    kappa_site: float;   gamma_fit: float;      snr_db: float
    quality: float


def estimate_band_tstar(ampl_f, freqs, r_km, fmin, fmax, gamma_init):
    try:
        m = (freqs >= fmin) & (freqs <= fmax)
        f = freqs[m]
        A = ampl_f[m]
        if len(A) < 3:
            return 0.0, gamma_init, 0.0
        A = np.maximum(A, 1e-12)
        y = np.log(A)
        X = np.vstack([np.ones_like(f), -np.pi * f]).T
        w = np.ones_like(f)
        beta = np.array([0.0, 0.0])
        for _ in range(10):
            beta, _, _, _ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
            resid = y - X.dot(beta)
            s = 1.4826 * np.median(np.abs(resid)) + 1e-9
            w = 1.0 / np.maximum(1.0, np.abs(resid / (1.5 * s)))
        tstar = beta[1]
        yhat = X.dot(beta)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-9
        r2 = 1 - ss_res / ss_tot
        return tstar, gamma_init, max(0.0, r2)
    except Exception:
        return 0.0, gamma_init, 0.0


def coda_decay_proxy(x, fs, laps, band):
    try:
        a, b = band
        n = x.shape[0]
        t = np.arange(n) / fs
        m = (t >= laps[0]) & (t <= laps[1])
        if not np.any(m) or n < 64:
            return 0.0
        ba = _safe_butter_band(fs, a, b)
        y = x.copy()
        if ba[0] is not None:
            try:
                y = sps.filtfilt(ba[0], ba[1], y, method='gust')
            except Exception:
                pass
        analytic = sps.hilbert(y)
        env = np.abs(analytic) + 1e-12
        # Guard against NaN from Hilbert
        if not np.all(np.isfinite(env)):
            env = np.nan_to_num(env, nan=1e-12, posinf=1e-12, neginf=1e-12)
        tt = t[m]
        yy = np.log(env[m])
        if len(tt) < 8:
            return 0.0
        X = np.vstack([np.ones_like(tt), -tt]).T
        beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
        slope = beta[1]
        return max(0.0, slope / np.pi)
    except Exception:
        return 0.0


def compute_path_features(fp_list, bands, cfg):
    fp = _pick_preferred_file(fp_list)
    if fp is None:
        return {}
    x, fs, sac = fast_read_sac(fp)
    if x is None or fs is None or len(x) < 100:
        return {}

    # Resample to target_fs
    if fs and fs > 1.75 * cfg.target_fs:
        ratio = fs / cfg.target_fs
        ir = int(round(ratio))
        try:
            if ir >= 2 and abs(ratio - ir) < 0.05:
                x_work = x
                k = ir
                for fct in (8, 4, 3, 2):
                    while k % fct == 0:
                        x_work = decimate(x_work, fct, ftype="iir",
                                          zero_phase=True)
                        k //= fct
                if k == 1:
                    x = x_work.astype(np.float32, copy=False)
                    fs = fs / ir
                else:
                    x = resample_poly(x, int(cfg.target_fs),
                                      int(fs)).astype(np.float32, copy=False)
                    fs = cfg.target_fs
            else:
                x = resample_poly(x, int(cfg.target_fs),
                                  int(fs)).astype(np.float32, copy=False)
                fs = cfg.target_fs
        except Exception:
            x = resample_poly(x, int(cfg.target_fs),
                              int(fs)).astype(np.float32, copy=False)
            fs = cfg.target_fs

    # SNR gate
    if cfg.min_snr_db > -999:
        try:
            y = _apply_band(x, fs, 2.0, 12.0)
            s = float(np.percentile(np.abs(y), 95))
            n_ = float(np.median(np.abs(y)) + 1e-9)
            snr_db_global = 20 * np.log10((s + 1e-6) / (n_ + 1e-6))
            if snr_db_global < cfg.min_snr_db:
                return {}
        except Exception:
            pass

    tr_proc = Trace(data=x.copy())
    tr_proc.stats.sampling_rate = fs
    pick_s = pick_onset(tr_proc, 'S') or 1.0
    post_eff = cfg.short_post_pick if cfg.fast_mode else cfg.post_pick
    coda_lps = cfg.coda_lapses if not cfg.fast_mode else [(30.0, 35.0)]
    dw, coda_ws = make_windows(pick_s, fs, cfg.pre_pick, post_eff, coda_lps)

    n = len(x)
    t = np.arange(n) / fs
    dm = (t >= dw[0]) & (t <= dw[1])
    xdir = x[dm] if np.any(dm) else x
    if len(xdir) < 64:
        return {}

    # FFT
    seg = xdir
    L = len(seg)
    if L < 2:
        return {}
    n_fast = next_fast_len(L)
    try:
        A = np.abs(sp_rfft(seg, n=n_fast, workers=os.cpu_count()))
        freqs = sp_rfftfreq(n_fast, d=1.0 / fs)
    except Exception:
        A = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)

    # Distance
    try:
        evlo = float(getattr(sac, "evlo", np.nan))
        evla = float(getattr(sac, "evla", np.nan))
        stlo = float(getattr(sac, "stlo", np.nan))
        stla = float(getattr(sac, "stla", np.nan))
        r_km = haversine_km(evlo, evla, stlo, stla)
    except Exception:
        r_km = 100.0

    # Kappa
    kappa_band = (min(10.0, 0.2 * fs), min(25.0, 0.45 * fs))
    try:
        kappa, _, _ = estimate_band_tstar(A, freqs, r_km,
                                          *kappa_band, cfg.geometric_gamma_init)
    except Exception:
        kappa = 0.0

    out = {}
    for bi, (fmin, fmax) in enumerate(bands):
        fmin_eff = max(0.05, min(fmin, 0.45 * fs))
        fmax_eff = max(fmin_eff + 0.05, min(fmax, 0.45 * fs))
        if fmin_eff >= fmax_eff:
            continue
        try:
            tstar, gamma_fit, r2 = estimate_band_tstar(
                A, freqs, r_km, fmin_eff, fmax_eff, cfg.geometric_gamma_init
            )
        except Exception:
            continue

        if cfg.fast_mode:
            ti = ts = 0.0
        else:
            coda_vals = []
            for laps in coda_ws:
                val = coda_decay_proxy(x, fs, laps, (fmin_eff, fmax_eff))
                if not np.isnan(val):
                    coda_vals.append(val)
            t_coda = float(np.median(coda_vals)) if coda_vals else 0.0
            if len(coda_ws) >= 2:
                ti = coda_decay_proxy(x, fs, coda_ws[-1], (fmin_eff, fmax_eff))
                ts = coda_decay_proxy(x, fs, coda_ws[0],  (fmin_eff, fmax_eff))
            else:
                ti, ts = t_coda, t_coda

        try:
            band_s = band_snr_db(xdir, fs, fmin_eff, fmax_eff)
        except Exception:
            band_s = 0.0

        out[bi] = PathBandFeatures(
            tstar_dir=float(max(0.0, tstar)),
            tstar_coda_i=float(max(0.0, ti if not np.isnan(ti) else 0.0)),
            tstar_coda_s=float(max(0.0, ts if not np.isnan(ts) else 0.0)),
            kappa_site=float(max(0.0, kappa)),
            gamma_fit=float(cfg.geometric_gamma_init),
            snr_db=float(band_s),
            quality=float(max(0.0, r2)),
        )
    return out


# ──────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION (threaded)
# ──────────────────────────────────────────────────────────────────────
def _extract_one(pf_key_fps_cfg):
    (event_id, station), fps, cfg_local = pf_key_fps_cfg
    try:
        band_feats = compute_path_features(fps, cfg_local.bands, cfg_local)
        return (event_id, station, band_feats) if band_feats else None
    except Exception as e:
        if cfg_local.debug_feature_extraction:
            print(f"Error processing {event_id}-{station}: {e}")
        return None


def _run_feature_extraction_with_threads(tasks, desc):
    all_feats = {}
    max_workers = max(2, os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_extract_one, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
            res = fut.result()
            if res is not None:
                ev, st, feats = res
                all_feats[(ev, st)] = feats
    return all_feats


def run_feature_extraction():
    print("[1/7] Feature extraction per path (parallel)...")
    tasks = []
    for pf in PATHS:
        key = (pf.event_id, pf.station)
        fps = FILE_INDEX.get(key, [])
        if fps:
            tasks.append((key, fps, cfg))
    all_feats = _run_feature_extraction_with_threads(tasks,
                                                     desc="Paths (threads)")
    np.save(os.path.join(cfg.out_dir, 'path_features.npy'),
            all_feats, allow_pickle=True)
    print(f"Saved features for {len(all_feats)} paths")
    return all_feats


# ══════════════════════════════════════════════════════════════════════
#  3-D GRID
# ══════════════════════════════════════════════════════════════════════
class Grid:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.xs = np.linspace(cfg.min_lon, cfg.max_lon, cfg.nx)
        self.ys = np.linspace(cfg.min_lat, cfg.max_lat, cfg.ny)
        self.zs = np.linspace(cfg.min_depth_km, cfg.max_depth_km, cfg.nz)
        self.dx = (cfg.max_lon - cfg.min_lon) / max(1, cfg.nx - 1)
        self.dy = (cfg.max_lat - cfg.min_lat) / max(1, cfg.ny - 1)
        self.dz = (cfg.max_depth_km - cfg.min_depth_km) / max(1, cfg.nz - 1)
        print(f"Grid: {cfg.nx}x{cfg.ny}x{cfg.nz} | "
              f"lon [{cfg.min_lon:.1f}, {cfg.max_lon:.1f}] | "
              f"lat [{cfg.min_lat:.1f}, {cfg.max_lat:.1f}] | "
              f"depth [{cfg.min_depth_km:.1f}, {cfg.max_depth_km:.1f}] km | "
              f"dz={self.dz:.2f} km")

    def voxel_index(self, lon, lat, dep_km):
        ix = int((lon - self.cfg.min_lon) / max(1e-9, self.dx))
        iy = int((lat - self.cfg.min_lat) / max(1e-9, self.dy))
        iz = int((dep_km - self.cfg.min_depth_km) / max(1e-9, self.dz))
        if (ix < 0 or iy < 0 or iz < 0
                or ix >= self.cfg.nx or iy >= self.cfg.ny
                or iz >= self.cfg.nz):
            return None
        return iz * (self.cfg.nx * self.cfg.ny) + iy * self.cfg.nx + ix

    def ray_voxel_lengths(self, lon1, lat1, dep1, lon2, lat2, dep2):
        nseg = 400
        t = np.linspace(0, 1, nseg)
        lons = lon1 + (lon2 - lon1) * t
        lats = lat1 + (lat2 - lat1) * t
        deps = dep1 + (dep2 - dep1) * t
        acc = {}
        for i in range(nseg - 1):
            idx = self.voxel_index(lons[i], lats[i], deps[i])
            if idx is None:
                continue
            dh = haversine_km(lons[i], lats[i], lons[i + 1], lats[i + 1])
            dv = abs(deps[i + 1] - deps[i])
            ds = (dh ** 2 + dv ** 2) ** 0.5
            acc[idx] = acc.get(idx, 0.0) + ds
        return acc


GRID = Grid(cfg)


def build_paths_data(all_feats):
    print('\n[2/7] Building path matrices...')
    paths_data = []
    for pf in tqdm(PATHS, desc='rays'):
        key = (pf.event_id, pf.station)
        if key not in all_feats:
            continue
        L = GRID.ray_voxel_lengths(
            pf.lon_e, pf.lat_e, max(0.0, pf.dep_e_km),
            pf.lon_s, pf.lat_s, 0.0,
        )
        obs = {}
        quality_weight = 0.0
        for bi in range(len(cfg.bands)):
            fb = all_feats[key].get(bi, None)
            if fb is None:
                continue
            obs[bi] = (fb.tstar_dir, fb.tstar_coda_i, fb.tstar_coda_s)
            quality_weight += (0.5 * fb.quality
                               + 0.5 * max(0.0, min(1.0,
                                   (fb.snr_db - cfg.min_snr_db) / 10.0)))
        if len(obs) == 0 or len(L) == 0:
            continue
        quality_weight = quality_weight / max(1, len(obs))
        paths_data.append({
            'L': L, 'obs': obs,
            'w': float(max(0.05, min(1.0, quality_weight))),
        })
    print(f"Paths used for inversion: {len(paths_data)}")
    np.save(os.path.join(cfg.out_dir, 'paths_data.npy'),
            paths_data, allow_pickle=True)
    return paths_data


# ──────────────────────────────────────────────────────────────────────
#  COVERAGE
# ──────────────────────────────────────────────────────────────────────
def compute_coverage_by_band(paths_data):
    nvox = cfg.nx * cfg.ny * cfg.nz
    B = len(cfg.bands)
    cov = [np.zeros(nvox, dtype=np.float32) for _ in range(B)]
    for pd in paths_data:
        L = pd['L']
        for b in range(B):
            if b not in pd['obs']:
                continue
            for v, seglen in L.items():
                if 0 <= v < nvox:
                    cov[b][v] += float(seglen)
    return tuple(cov)


def coverage_histogram_global(cov_tuple):
    return np.sum(np.vstack(cov_tuple), axis=0)


# ══════════════════════════════════════════════════════════════════════
#  TORCH-DEPENDENT PARTS
# ══════════════════════════════════════════════════════════════════════
def _define_gpu_parts():
    torch, nn, F = _lazy_import_torch_stack()

    def get_device():
        print(f"PyTorch {torch.__version__} | CUDA build: {torch.version.cuda}")
        if not torch.cuda.is_available():
            print("CUDA not available; using CPU.")
            return torch.device("cpu")
        n = torch.cuda.device_count()
        print(f"Found {n} CUDA device(s):")
        for i in range(n):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device

    class DnLite(nn.Module):
        def __init__(self, nvox, p_drop=0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 9, padding=4), nn.ReLU(inplace=True),
                nn.Dropout(p_drop),
                nn.Conv1d(32, 32, 9, padding=4), nn.ReLU(inplace=True),
                nn.Dropout(p_drop),
                nn.Conv1d(32, 1, 9, padding=4),
            )

        def forward(self, x):
            return x - self.net(x.unsqueeze(1)).squeeze(1)

    def build_L_sparse(paths_data, n_vox, device, dtype):
        rows, cols, vals = [], [], []
        P = len(paths_data)
        for p, pd in enumerate(paths_data):
            for v, L in pd['L'].items():
                if 0 <= v < n_vox:
                    rows.append(p)
                    cols.append(v)
                    vals.append(L)
        if len(rows) == 0:
            crow = torch.tensor([0] * (P + 1), dtype=torch.int64,
                                device=device)
            return torch.sparse_csr_tensor(
                crow,
                torch.tensor([], dtype=torch.int64, device=device),
                torch.tensor([], dtype=dtype, device=device),
                size=(P, n_vox), dtype=dtype, device=device,
            )
        # Sort by (row, col) for correct CSR construction
        order = sorted(range(len(rows)), key=lambda i: (rows[i], cols[i]))
        rows_s = [rows[i] for i in order]
        cols_s = [cols[i] for i in order]
        vals_s = [vals[i] for i in order]

        rows_t = torch.tensor(rows_s, dtype=torch.int64, device=device)
        cols_t = torch.tensor(cols_s, dtype=torch.int64, device=device)
        vals_t = torch.tensor(vals_s, dtype=dtype, device=device)
        crow = torch.zeros(P + 1, dtype=torch.int64, device=device)
        crow.index_add_(0, rows_t + 1,
                        torch.ones_like(rows_t, dtype=torch.int64))
        crow = torch.cumsum(crow, dim=0)
        return torch.sparse_csr_tensor(crow, cols_t, vals_t,
                                       size=(P, n_vox), dtype=dtype,
                                       device=device)

    class AttenInversion(nn.Module):
        def __init__(self, grid, n_bands, lambda_tv, lambda_aniso,
                     lambda_pnp, pnp_noise_std, p_drop,
                     faults=None, device=torch.device('cpu'),
                     param_dtype=torch.float32, L_csr=None):
            super().__init__()
            self.grid = grid
            self.nvox = grid.cfg.nx * grid.cfg.ny * grid.cfg.nz
            self.n_bands = n_bands
            self.lambda_tv = lambda_tv
            self.lambda_aniso = lambda_aniso
            self.lambda_pnp = lambda_pnp
            self.pnp_noise_std = pnp_noise_std
            self.Qi = nn.Parameter(
                torch.zeros(n_bands, self.nvox, device=device,
                            dtype=param_dtype)
            )
            self.Qs = nn.Parameter(
                torch.zeros(n_bands, self.nvox, device=device,
                            dtype=param_dtype)
            )
            self.denoisers = nn.ModuleList(
                [DnLite(self.nvox, p_drop) for _ in range(n_bands)]
            )
            self.denoisers.to(device)
            self.fault_edges = self._build_anisotropy_edges(faults)
            assert L_csr is not None
            self.L = L_csr
            self.device = device

        def _build_anisotropy_edges(self, faults):
            edges = []
            if not faults:
                return edges
            xv, yv = np.meshgrid(self.grid.xs, self.grid.ys, indexing='xy')
            tree = STRtree(
                [Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())]
            )
            for poly in faults:
                idxs = [tree.nearest(Point(lon, lat))._geom_idx
                        for lon, lat in poly]
                for a, b in zip(idxs[:-1], idxs[1:]):
                    edges.append((a, b))
            return edges

        def tv_loss(self, X):
            nx, ny, nz = (self.grid.cfg.nx, self.grid.cfg.ny,
                          self.grid.cfg.nz)
            Xv = X.view(self.n_bands, nz, ny, nx)
            dx = torch.abs(Xv[:, :, :, 1:] - Xv[:, :, :, :-1]).mean()
            dy = torch.abs(Xv[:, :, 1:, :] - Xv[:, :, :-1, :]).mean()
            dz = (torch.abs(Xv[:, 1:, :, :] - Xv[:, :-1, :, :]).mean()
                  if nz > 1 else Xv.new_zeros(()))
            return dx + dy + dz

        def aniso_loss(self, X):
            if not self.fault_edges:
                return X.new_zeros(())
            nx, ny = self.grid.cfg.nx, self.grid.cfg.ny
            X2D = X.view(self.n_bands, self.grid.cfg.nz, ny, nx).mean(dim=1)
            loss = X2D.new_zeros(())
            for (a, b) in self.fault_edges:
                ya, xa = divmod(a, nx)
                yb, xb = divmod(b, nx)
                if (0 <= ya < ny and 0 <= xa < nx
                        and 0 <= yb < ny and 0 <= xb < nx):
                    loss = loss + torch.abs(
                        X2D[:, ya, xa] - X2D[:, yb, xb]
                    ).mean()
            return loss / (len(self.fault_edges) + 1e-6)

        def pnp_loss(self, Q):
            if self.lambda_pnp <= 0.0:
                return Q.new_zeros(())
            loss = Q.new_zeros(())
            for b in range(self.n_bands):
                z = Q[b] + self.pnp_noise_std * torch.randn_like(Q[b])
                Dz = self.denoisers[b](z.unsqueeze(0)).squeeze(0)
                loss = loss + torch.mean((Q[b] - Dz) ** 2)
            return loss / self.n_bands

        def forward_spmm(self, obs_dir, obs_i, obs_s, w_path,
                         weights_ab=(0.6, 0.4), compute_reg=True):
            a, b = weights_ab
            qeff = a * self.Qi + b * self.Qs
            pred   = (self.L @ qeff.T).T
            pred_i = (self.L @ self.Qi.T).T
            pred_s = (self.L @ self.Qs.T).T
            data = ((pred - obs_dir) ** 2
                    + 0.3 * (pred_i - obs_i) ** 2
                    + 0.3 * (pred_s - obs_s) ** 2)
            data = (data * w_path.unsqueeze(0)).mean()

            if compute_reg:
                tv = self.tv_loss(self.Qi) + self.tv_loss(self.Qs)
                aniso = self.aniso_loss(self.Qi) + self.aniso_loss(self.Qs)
                pnp = self.pnp_loss(self.Qi) + self.pnp_loss(self.Qs)
            else:
                tv = data.new_zeros(())
                aniso = data.new_zeros(())
                pnp = data.new_zeros(())

            total = (data + self.lambda_tv * tv
                     + self.lambda_aniso * aniso
                     + self.lambda_pnp * pnp)
            return total, {
                "data": float(data.detach()),
                "tv": float(tv.detach()),
                "aniso": float(aniso.detach()),
                "pnp": float(pnp.detach()),
                "pred": pred.detach(),
                "pred_i": pred_i.detach(),
                "pred_s": pred_s.detach(),
            }

        @torch.no_grad()
        def mc_prior_samples(self, n_samples=4):
            if self.lambda_pnp <= 0.0 or n_samples <= 0:
                return None
            out_Qi, out_Qs = [], []
            for _ in range(n_samples):
                for D in self.denoisers:
                    D.train()
                Qi_s = torch.stack([
                    self.denoisers[b](self.Qi[b].unsqueeze(0)).squeeze(0)
                    for b in range(self.n_bands)
                ])
                Qs_s = torch.stack([
                    self.denoisers[b](self.Qs[b].unsqueeze(0)).squeeze(0)
                    for b in range(self.n_bands)
                ])
                out_Qi.append(Qi_s.clone())
                out_Qs.append(Qs_s.clone())
            return torch.stack(out_Qi), torch.stack(out_Qs)

    def run_inversion(paths_data, member_idx=0, seed=123,
                      bootstrap_frac=1.0):
        print(f'\n[3/7] Inversion member {member_idx + 1} '
              f'(bootstrap={bootstrap_frac:.2f})...')
        import torch as torch_local
        _set_seed(seed + 17 * member_idx)
        device = get_device()
        use_cuda = (device.type == 'cuda')
        param_dtype = torch_local.float32

        Ptot = len(paths_data)
        Psel = max(1, int(round(bootstrap_frac * Ptot)))
        sel_idx = np.sort(
            np.random.choice(np.arange(Ptot), size=Psel, replace=False)
        )
        pd_sel = [paths_data[i] for i in sel_idx]

        B = len(cfg.bands)
        P = len(pd_sel)
        obs_dir = torch_local.zeros((B, P), dtype=param_dtype, device=device)
        obs_i   = torch_local.zeros((B, P), dtype=param_dtype, device=device)
        obs_s   = torch_local.zeros((B, P), dtype=param_dtype, device=device)
        w_path  = torch_local.zeros((P,),   dtype=param_dtype, device=device)
        for p, pd in enumerate(pd_sel):
            for b in range(B):
                if b in pd['obs']:
                    od, oi, os_ = pd['obs'][b]
                    obs_dir[b, p] = float(od)
                    obs_i[b, p] = float(oi)
                    obs_s[b, p] = float(os_)
            w_path[p] = float(pd['w'])

        L_csr = build_L_sparse(pd_sel, n_vox=cfg.nx * cfg.ny * cfg.nz,
                               device=device, dtype=param_dtype)

        faults = None
        if cfg.faults_geojson and os.path.exists(cfg.faults_geojson):
            try:
                gj = json.load(
                    open(cfg.faults_geojson, 'r', encoding="utf-8")
                )
                faults = []
                for feat in gj['features']:
                    coords = feat['geometry']['coordinates']
                    if feat['geometry']['type'] == 'LineString':
                        faults.append([(lon, lat) for lon, lat in coords])
                    elif feat['geometry']['type'] == 'MultiLineString':
                        for ls in coords:
                            faults.append([(lon, lat) for lon, lat in ls])
            except Exception as e:
                print('fault geojson read error:', e)

        inv = AttenInversion(
            GRID, n_bands=B,
            lambda_tv=cfg.lambda_tv, lambda_aniso=cfg.lambda_aniso,
            lambda_pnp=cfg.lambda_pnp, pnp_noise_std=cfg.pnp_noise_std,
            p_drop=cfg.pnp_dropout_p,
            faults=faults, device=device,
            param_dtype=param_dtype, L_csr=L_csr,
        )

        max_iter = min(cfg.max_iter_inv, 300)
        lr = cfg.lr_inv * (2.0 if use_cuda else 1.0)
        opt = torch_local.optim.Adam(
            [inv.Qi, inv.Qs] + list(inv.denoisers.parameters()), lr=lr
        )
        sched = torch_local.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(50, max_iter)
        )

        compute_reg_every = 5
        tol = 1e-4
        patience = 10
        best = float('inf')
        stagn = 0
        losses_hist = []

        bar = tqdm(range(max_iter), desc=f'inv iters m{member_idx + 1}')
        for it in bar:
            do_reg = (it % compute_reg_every == 0)
            total, parts = inv.forward_spmm(
                obs_dir, obs_i, obs_s, w_path, compute_reg=do_reg
            )
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            sched.step()

            if it % 5 == 0:
                with torch_local.no_grad():
                    full_total, full_parts = inv.forward_spmm(
                        obs_dir, obs_i, obs_s, w_path, compute_reg=True
                    )
                cur = float(full_total.detach().cpu())
                losses_hist.append((
                    it, cur, full_parts["data"], full_parts["tv"],
                    full_parts["aniso"], full_parts["pnp"],
                ))
                bar.set_postfix({"loss": f"{cur:.6f}"})
                if cur < best * (1 - tol):
                    best = cur
                    stagn = 0
                else:
                    stagn += 1
                    if stagn >= patience:
                        print(f"Early stop at iter {it} | best={best:.6f}")
                        break

        # Save loss CSV
        loss_csv = os.path.join(cfg.out_dir,
                                f"inversion_loss_m{member_idx + 1}.csv")
        with open(loss_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["iter", "total", "data", "tv", "aniso", "pnp"])
            for r in losses_hist:
                w.writerow(list(r))

        # ── Enhanced learning-curve plot ──
        if losses_hist:
            _plot_learning_curve(losses_hist, member_idx)

        Qi = inv.Qi.detach().cpu().numpy()
        Qs = inv.Qs.detach().cpu().numpy()

        with torch_local.no_grad():
            _, parts_final = inv.forward_spmm(
                obs_dir, obs_i, obs_s, w_path, compute_reg=True
            )
            pred  = parts_final["pred"].cpu().numpy()
            predi = parts_final["pred_i"].cpu().numpy()
            preds = parts_final["pred_s"].cpu().numpy()

        np.save(os.path.join(cfg.out_dir,
                             f'pred_total_m{member_idx + 1}.npy'), pred)
        np.save(os.path.join(cfg.out_dir,
                             f'pred_i_m{member_idx + 1}.npy'), predi)
        np.save(os.path.join(cfg.out_dir,
                             f'pred_s_m{member_idx + 1}.npy'), preds)

        mc_Qi = []
        mc_Qs = []
        if cfg.lambda_pnp > 0.0 and cfg.mc_prior_samples > 0:
            mc = inv.mc_prior_samples(n_samples=cfg.mc_prior_samples)
            if mc is not None:
                mQi, mQs = mc
                mc_Qi = mQi.cpu().numpy()
                mc_Qs = mQs.cpu().numpy()

        out = {
            "Qi": Qi, "Qs": Qs,
            "pred_total": pred, "pred_i": predi, "pred_s": preds,
            "mc_Qi": mc_Qi, "mc_Qs": mc_Qs,
        }
        if use_cuda:
            torch_local.cuda.empty_cache()
        return out

    return run_inversion


# ══════════════════════════════════════════════════════════════════════
#  ENHANCED PLOTTING & EXPORT
# ══════════════════════════════════════════════════════════════════════

def _add_geo_grid(ax, extent):
    """Add geographic grid lines and degree labels."""
    lon0, lon1, lat0, lat1 = extent
    ax.set_xlabel("Longitude (°E)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=14, fontweight="bold")
    # Lon ticks
    lon_ticks = np.arange(np.ceil(lon0), lon1 + 0.01, 1.0)
    lat_ticks = np.arange(np.ceil(lat0), lat1 + 0.01, 1.0)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    # Light grid
    for lt in lat_ticks:
        ax.axhline(lt, color='#999999', linewidth=0.3, alpha=0.4)
    for ln in lon_ticks:
        ax.axvline(ln, color='#999999', linewidth=0.3, alpha=0.4)


def _smart_colorbar(im, ax, label, fig=None):
    """Create a well-formatted colorbar."""
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    cbar.ax.yaxis.set_major_formatter(fmt)
    return cbar


def _plot_learning_curve(losses_hist, member_idx):
    """Enhanced learning curve with dual axes."""
    its = [r[0] for r in losses_hist]
    totv = [r[1] for r in losses_hist]
    datv = [r[2] for r in losses_hist]
    tvv  = [r[3] for r in losses_hist]
    aniv = [r[4] for r in losses_hist]
    pnpv = [r[5] for r in losses_hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: total + data
    ax1.plot(its, totv, color="#e63946", linewidth=2.8,
             label="Total", zorder=5)
    ax1.plot(its, datv, color="#457b9d", linewidth=2.2,
             label="Data", linestyle="--")
    ax1.fill_between(its, datv, totv, alpha=0.12, color="#e63946")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Convergence — Member {member_idx + 1}",
                  fontsize=16, fontweight="bold")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())

    # Right: regularization terms
    ax2.plot(its, tvv, color="#2a9d8f", linewidth=2.2, label="TV")
    ax2.plot(its, aniv, color="#e9c46a", linewidth=2.2, label="Aniso")
    ax2.plot(its, pnpv, color="#a855f7", linewidth=2.2, label="PnP")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Regularization Loss")
    ax2.set_title("Regularization Terms")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(cfg.out_dir,
                             f"inversion_loss_m{member_idx + 1}.png"))
    plt.close(fig)


def _plot_maps(Q, name_prefix):
    """Enhanced map slices with contours, geo-grid, and vibrant colors."""
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    extent = [cfg.min_lon, cfg.max_lon, cfg.min_lat, cfg.max_lat]

    # Select colormap
    if name_prefix.startswith("Qi") and "std" in name_prefix:
        cmap = CMAP_STD
    elif name_prefix.startswith("Qi"):
        cmap = CMAP_QI
    elif "std" in name_prefix:
        cmap = CMAP_STD
    elif name_prefix.startswith("Qsct"):
        cmap = CMAP_QSCT
    else:
        cmap = CMAP_SCAT

    for bi in range(Q.shape[0]):
        Q_b = Q[bi].reshape(nz, ny, nx)
        for iz in range(nz):
            data = Q_b[iz]
            fig, ax = plt.subplots(figsize=cfg.map_figsize)

            # Robust percentile scaling for contrast
            finite = data[np.isfinite(data)]
            if finite.size > 0:
                vmin = np.percentile(finite, 1)
                vmax = np.percentile(finite, 99)
                if abs(vmax - vmin) < 1e-12:
                    vmin, vmax = finite.min(), finite.max()
            else:
                vmin, vmax = 0, 1

            im = ax.imshow(
                data, origin='lower', extent=extent, cmap=cmap,
                aspect="auto", vmin=vmin, vmax=vmax,
                interpolation="bilinear",
            )

            # Contour overlay
            try:
                lon_c = np.linspace(extent[0], extent[1], nx)
                lat_c = np.linspace(extent[2], extent[3], ny)
                LON, LAT = np.meshgrid(lon_c, lat_c)
                levels = np.linspace(vmin, vmax, cfg.contour_levels)
                cs = ax.contour(
                    LON, LAT, data, levels=levels, colors='white',
                    linewidths=0.5, alpha=0.45,
                )
                ax.clabel(cs, inline=True, fontsize=7, fmt='%.3g')
            except Exception:
                pass

            # Subregion outlines
            for reg_name, verts in cfg.subregions.items():
                poly_x = [v[0] for v in verts] + [verts[0][0]]
                poly_y = [v[1] for v in verts] + [verts[0][1]]
                ax.plot(poly_x, poly_y, color='#ffffff', linewidth=1.5,
                        linestyle='--', alpha=0.7,
                        path_effects=[
                            pe.Stroke(linewidth=2.8, foreground='black',
                                      alpha=0.5),
                            pe.Normal(),
                        ])

            _add_geo_grid(ax, extent)

            label_txt = (f'{name_prefix}$^{{-1}}$'
                         if name_prefix in ("Qi", "Qsct")
                         else name_prefix)
            _smart_colorbar(im, ax, label_txt)

            band_lo, band_hi = cfg.bands[bi]
            ax.set_title(
                f'{label_txt}  ·  {band_lo}–{band_hi} Hz  ·  '
                f'z = {GRID.zs[iz]:.1f} km',
                fontsize=16, fontweight="bold",
            )

            fig.tight_layout()
            fig.savefig(os.path.join(
                cfg.out_dir, f'{name_prefix}_b{bi}_z{iz}.png'
            ))
            plt.close(fig)


def _plot_cross_sections(Q, name_prefix):
    """Depth cross-sections: lon–depth at mid-lat, lat–depth at mid-lon."""
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    if nz < 2:
        return

    if name_prefix.startswith("Qi") and "std" not in name_prefix:
        cmap = CMAP_QI
    elif "std" in name_prefix:
        cmap = CMAP_STD
    elif name_prefix.startswith("Qsct"):
        cmap = CMAP_QSCT
    else:
        cmap = CMAP_SCAT

    for bi in range(Q.shape[0]):
        Q_b = Q[bi].reshape(nz, ny, nx)
        band_lo, band_hi = cfg.bands[bi]
        label_txt = (f'{name_prefix}$^{{-1}}$'
                     if name_prefix in ("Qi", "Qsct") else name_prefix)

        # ── Lon–Depth cross-section at mid-latitude ──
        mid_iy = ny // 2
        slice_xz = Q_b[:, mid_iy, :]  # shape (nz, nx)
        finite = slice_xz[np.isfinite(slice_xz)]
        if finite.size == 0:
            continue
        vmin, vmax = np.percentile(finite, 2), np.percentile(finite, 98)
        if abs(vmax - vmin) < 1e-12:
            vmin, vmax = finite.min(), finite.max()

        fig, ax = plt.subplots(figsize=cfg.cross_section_figsize)
        im = ax.imshow(
            slice_xz, origin='upper', aspect='auto',
            extent=[cfg.min_lon, cfg.max_lon,
                    cfg.max_depth_km, cfg.min_depth_km],
            cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="bilinear",
        )
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Depth (km)")
        ax.set_title(
            f'{label_txt}  ·  {band_lo}–{band_hi} Hz  ·  '
            f'Lat = {GRID.ys[mid_iy]:.1f}°N (cross-section)',
            fontsize=15, fontweight="bold",
        )
        _smart_colorbar(im, ax, label_txt)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        fig.tight_layout()
        fig.savefig(os.path.join(
            cfg.out_dir,
            f'{name_prefix}_xsec_lon_b{bi}_lat{mid_iy}.png'
        ))
        plt.close(fig)

        # ── Lat–Depth cross-section at mid-longitude ──
        mid_ix = nx // 2
        slice_yz = Q_b[:, :, mid_ix]  # shape (nz, ny)

        fig, ax = plt.subplots(figsize=cfg.cross_section_figsize)
        im = ax.imshow(
            slice_yz, origin='upper', aspect='auto',
            extent=[cfg.min_lat, cfg.max_lat,
                    cfg.max_depth_km, cfg.min_depth_km],
            cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="bilinear",
        )
        ax.set_xlabel("Latitude (°N)")
        ax.set_ylabel("Depth (km)")
        ax.set_title(
            f'{label_txt}  ·  {band_lo}–{band_hi} Hz  ·  '
            f'Lon = {GRID.xs[mid_ix]:.1f}°E (cross-section)',
            fontsize=15, fontweight="bold",
        )
        _smart_colorbar(im, ax, label_txt)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        fig.tight_layout()
        fig.savefig(os.path.join(
            cfg.out_dir,
            f'{name_prefix}_xsec_lat_b{bi}_lon{mid_ix}.png'
        ))
        plt.close(fig)


def _save_geotiff_slices(Q, name_prefix):
    try:
        import rasterio
        from rasterio.transform import from_origin
    except Exception as e:
        print("GeoTIFF export skipped (rasterio missing):", e)
        return []
    stats_rows = []
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    dx = (cfg.max_lon - cfg.min_lon) / max(1, nx - 1)
    dy = (cfg.max_lat - cfg.min_lat) / max(1, ny - 1)
    transform = from_origin(cfg.min_lon, cfg.max_lat, dx, dy)
    for bi in range(Q.shape[0]):
        Q_b = Q[bi].reshape(nz, ny, nx)
        for iz in range(nz):
            # Flip for north-up GeoTIFF convention
            arr = np.flipud(Q_b[iz].astype(np.float32))
            out_fp = os.path.join(
                cfg.out_dir,
                f"{name_prefix}_b{bi}_z{iz}_depth_"
                f"{GRID.zs[iz]:.1f}km.tif",
            )
            with rasterio.open(
                out_fp, "w", driver="GTiff",
                height=arr.shape[0], width=arr.shape[1],
                count=1, dtype=arr.dtype, crs="EPSG:4326",
                transform=transform, nodata=np.nan, compress="lzw",
            ) as dst:
                dst.write(arr, 1)
            nan_frac = float(np.isnan(arr).mean())
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                vmin = vmax = mean = std = np.nan
            else:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
                mean = float(np.nanmean(finite))
                std  = float(np.nanstd(finite))
            extreme = (std > 0 and
                       (vmax > mean + 5 * std or vmin < mean - 5 * std))
            all_nan = nan_frac >= 0.9
            flag = ("EXTREME" if extreme
                    else ("ALL_NAN" if all_nan else "OK"))
            stats_rows.append([
                name_prefix, bi, iz, float(GRID.zs[iz]),
                vmin, vmax, mean, std, nan_frac, flag,
            ])
    return stats_rows


def _save_voxel_csv(Q, name_prefix, filename):
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    with open(os.path.join(cfg.out_dir, filename), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band_index", "lon", "lat", "depth_km", "value"])
        for bi in range(Q.shape[0]):
            Q_b = Q[bi].reshape(nz, ny, nx)
            for iz, z in enumerate(GRID.zs):
                for iy, lat in enumerate(GRID.ys):
                    for ix, lon in enumerate(GRID.xs):
                        w.writerow([bi, lon, lat, float(z),
                                    float(Q_b[iz, iy, ix])])


def _band_centers_hz():
    return np.array([math.sqrt(lo * hi) for (lo, hi) in cfg.bands],
                    dtype=float)


def _qinv_vs_freq_errorbars(Q_mean, Q_std, title_prefix="Qi"):
    """Q⁻¹(f) curves with shaded uncertainty bands."""
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    fcs = _band_centers_hz()
    B = len(cfg.bands)

    # Select depths to plot (at most 6 representative layers)
    if nz <= 6:
        plot_depths = list(range(nz))
    else:
        plot_depths = np.linspace(0, nz - 1, 6, dtype=int).tolist()

    means_per_depth = []
    stds_per_depth = []
    for iz in range(nz):
        m = []
        s = []
        for b in range(B):
            arr = Q_mean[b].reshape(nz, ny, nx)[iz]
            m.append(np.nanmean(arr))
            if Q_std is not None:
                s.append(np.nanmean(Q_std[b].reshape(nz, ny, nx)[iz]))
            else:
                s.append(0.0)
        means_per_depth.append(np.array(m))
        stds_per_depth.append(np.array(s))

    # Color cycle from a vivid colormap
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(plot_depths)))

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for ci, iz in enumerate(plot_depths):
        m = means_per_depth[iz]
        s = stds_per_depth[iz]
        c = colors[ci]
        label = f"{GRID.zs[iz]:.1f} km"
        ax.plot(fcs, m, '-o', color=c, linewidth=2.3, markersize=7,
                label=label, zorder=5)
        ax.fill_between(fcs, m - s, m + s, color=c, alpha=0.15)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"{title_prefix}$^{{-1}}$", fontsize=14, fontweight="bold")
    ax.set_title(
        f"{title_prefix}$^{{-1}}(f)$ with uncertainty — selected depths",
        fontsize=16, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Depth", title_fontsize=12, fontsize=11,
              loc="upper right", framealpha=0.92)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(os.path.join(
        cfg.out_dir, f"{title_prefix}_qinv_vs_freq_all_depths.png"
    ))
    plt.close(fig)


def _plot_residual_diagnostics(obs_dir, pred, band_labels, tag="total"):
    """Enhanced obs-vs-pred scatter + residual histogram with KDE."""
    B, P = obs_dir.shape
    for b in range(B):
        ot = obs_dir[b]
        pt = pred[b]
        res_t = ot - pt
        rmse_val = float(np.sqrt(np.mean(res_t ** 2)))
        corr = float(np.corrcoef(ot, pt)[0, 1]) if P > 2 else 0.0

        # ── Scatter ──
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        norm = TwoSlopeNorm(vmin=res_t.min(), vcenter=0,
                            vmax=res_t.max())
        sc = ax.scatter(
            ot, pt, s=18, alpha=0.7, c=res_t, cmap=CMAP_RESID,
            norm=norm, edgecolors="#333333", linewidths=0.35, zorder=4,
        )
        lims = [min(ot.min(), pt.min()), max(ot.max(), pt.max())]
        ax.plot(lims, lims, linewidth=2.5, color="#2a9d8f",
                label="1:1 line", zorder=3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Observed ({tag})")
        ax.set_ylabel(f"Predicted ({tag})")
        ax.set_title(f"Obs vs Pred — Band {band_labels[b]} ({tag})",
                     fontsize=15, fontweight="bold")

        # Stats box
        stats_text = (f"RMSE = {rmse_val:.4f}\n"
                      f"r = {corr:.3f}\n"
                      f"N = {P}")
        ax.text(
            0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffffffdd",
                      ec="#444444", linewidth=1.2),
        )
        ax.legend(loc="lower right")
        _smart_colorbar(sc, ax, "Residual")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(
            cfg.out_dir, f"obs_vs_pred_scatter_{tag}_band{b}.png"
        ))
        plt.close(fig)

        # ── Histogram + KDE ──
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        n_bins = min(60, max(20, P // 8))
        ax.hist(res_t, bins=n_bins, density=True, color="#3a0ca3",
                edgecolor="#1b1b2f", alpha=0.8, zorder=3,
                label="Residuals")
        try:
            kde = gaussian_kde(res_t)
            xs = np.linspace(res_t.min(), res_t.max(), 400)
            ax.plot(xs, kde(xs), linewidth=2.5, color="#f72585",
                    label="KDE", zorder=5)
            # Gaussian reference
            mu, sd = np.mean(res_t), np.std(res_t) + 1e-12
            gauss = (1.0 / (np.sqrt(2 * np.pi) * sd)
                     * np.exp(-0.5 * ((xs - mu) / sd) ** 2))
            ax.plot(xs, gauss, linewidth=2.0, color="#06d6a0",
                    linestyle="--", label="Gaussian ref", zorder=4)
        except Exception:
            pass
        ax.axvline(0, color='#e63946', linewidth=1.5, linestyle=':',
                   alpha=0.8)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.set_title(f"Residual Distribution — Band {band_labels[b]} ({tag})",
                     fontsize=15, fontweight="bold")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(
            cfg.out_dir, f"residual_hist_{tag}_band{b}.png"
        ))
        plt.close(fig)


def _plot_coverage_maps(cov_tuple):
    """Enhanced coverage maps with log-scale and contours."""
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    extent = [cfg.min_lon, cfg.max_lon, cfg.min_lat, cfg.max_lat]

    for b in range(len(cfg.bands)):
        cov3 = cov_tuple[b].reshape(nz, ny, nx)
        cov2 = np.nansum(cov3, axis=0)

        fig, ax = plt.subplots(figsize=cfg.map_figsize)
        # Log transform for better visibility
        cov2_plot = np.log1p(cov2)
        finite = cov2_plot[np.isfinite(cov2_plot)]
        if finite.size == 0:
            continue
        vmin = np.percentile(finite, 1)
        vmax = np.percentile(finite, 99)

        im = ax.imshow(
            cov2_plot, origin='lower', extent=extent, cmap=CMAP_COV,
            aspect="auto", vmin=vmin, vmax=vmax,
            interpolation="bilinear",
        )
        _add_geo_grid(ax, extent)
        _smart_colorbar(im, ax, "log₁₊(Coverage)")

        band_lo, band_hi = cfg.bands[b]
        ax.set_title(
            f"Path Density  ·  {band_lo}–{band_hi} Hz",
            fontsize=16, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.out_dir, f"coverage_band{b}.png"))
        plt.close(fig)

    # Global histogram
    all_cov = coverage_histogram_global(cov_tuple)
    vals = all_cov[all_cov > 0]
    if len(vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.hist(vals, bins=70, color="#4361ee", alpha=0.85,
            edgecolor="#1b1b2f", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Coverage (total ray-length per voxel)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Ray Coverage Distribution (all bands)",
                 fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, "coverage_histogram.png"))
    plt.close(fig)


def _coverage_vs_std_scatter(cov_tuple, Q_std_mean, tag="Qi"):
    """Coverage vs uncertainty with marginal histograms."""
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    std2 = np.nanmean(Q_std_mean[0].reshape(nz, ny, nx), axis=0)
    cov2 = np.nansum(cov_tuple[0].reshape(nz, ny, nx), axis=0)
    x = cov2.ravel()
    y = std2.ravel()
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if not np.any(m):
        return

    fig = plt.figure(figsize=(9.5, 7.5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_rt   = fig.add_subplot(gs[1, 1], sharey=ax_main)

    sc = ax_main.scatter(
        x[m], y[m], s=6, c=y[m], cmap=CMAP_QSCT, alpha=0.65,
        edgecolors="none",
    )
    ax_main.set_xlabel("Coverage")
    ax_main.set_ylabel(f"Std({tag}$^{{-1}}$)")
    ax_main.grid(True, alpha=0.2)

    ax_top.hist(x[m], bins=60, color="#457b9d", alpha=0.8,
                edgecolor="none")
    ax_top.set_ylabel("Count")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_rt.hist(y[m], bins=60, orientation='horizontal',
               color="#e63946", alpha=0.8, edgecolor="none")
    ax_rt.set_xlabel("Count")
    plt.setp(ax_rt.get_yticklabels(), visible=False)

    fig.suptitle(f"Coverage vs Uncertainty — {tag}",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(cfg.out_dir, f"coverage_vs_std_{tag}.png"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  SUBREGION STATISTICS — robust depth clamping
# ══════════════════════════════════════════════════════════════════════
def _region_stats(Q_mean, Q_std, name_prefix="Qi"):
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    if nz == 0 or GRID.dz <= 0:
        print(f"  Skipping {name_prefix} subregion stats (nz=0 or dz<=0)")
        return

    lon_grid, lat_grid = np.meshgrid(GRID.xs, GRID.ys, indexing="xy")
    results = []

    for reg_name, verts in cfg.subregions.items():
        poly = Polygon(verts)

        # Parse depth range from name
        z0, z1 = cfg.min_depth_km, cfg.max_depth_km
        try:
            parts = reg_name.split("_")
            nums = [p.replace("km", "") for p in parts
                    if p.replace("km", "").replace(".", "", 1).isdigit()]
            if len(nums) >= 2:
                z0 = float(nums[0])
                z1 = float(nums[1])
        except Exception:
            pass

        iz0 = int(round((z0 - cfg.min_depth_km) / GRID.dz))
        iz1 = int(round((z1 - cfg.min_depth_km) / GRID.dz))
        iz0 = max(0, min(iz0, nz - 1))
        iz1 = max(0, min(iz1, nz - 1))
        if iz0 > iz1:
            iz0, iz1 = iz1, iz0

        for b in range(Q_mean.shape[0]):
            Qm = Q_mean[b].reshape(nz, ny, nx)
            Qs = (Q_std[b].reshape(nz, ny, nx)
                  if Q_std is not None else None)
            vals = []
            sigs = []
            for iz in range(iz0, iz1 + 1):
                arr = Qm[iz]
                mask = np.array([
                    poly.contains(Point(lon_grid[iy, ix], lat_grid[iy, ix]))
                    for iy in range(ny) for ix in range(nx)
                ], dtype=bool).reshape(ny, nx)
                sel = arr[mask]
                if sel.size > 0:
                    vals.append(np.nanmean(sel))
                    if Qs is not None:
                        sigs.append(np.nanmean(Qs[iz][mask]))
            if len(vals) > 0:
                results.append([
                    reg_name, name_prefix, b,
                    float(np.nanmean(vals)),
                    float(np.nanmean(sigs)) if sigs else float('nan'),
                ])

    # Write CSV
    out_csv = os.path.join(cfg.out_dir, f"{name_prefix}_subregion_stats.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "field", "band", "mean", "std"])
        for r in results:
            w.writerow(r)

    # Write text summary
    out_txt = os.path.join(cfg.out_dir, f"{name_prefix}_subregion_stats.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Uncertainty-aware summaries for {name_prefix}\n")
        f.write(f"Grid depth range: {cfg.min_depth_km}–{cfg.max_depth_km} km "
                f"({nz} layers, dz={GRID.dz:.2f} km)\n\n")
        for r in results:
            region, field, band, mean, std = r
            lo, hi = cfg.bands[int(band)]
            f.write(f"{region} | {field}^-1 | band {lo}-{hi} Hz : "
                    f"{mean:.4g} +/- {std:.3g}\n")

    print(f"  Subregion stats: {len(results)} entries → {out_csv}")

    # ── Bar chart per band ──
    if results:
        _plot_subregion_bars(results, name_prefix)


def _plot_subregion_bars(results, name_prefix):
    """Grouped bar chart of subregion means ± std per band."""
    B = len(cfg.bands)
    regions = sorted(set(r[0] for r in results))
    if not regions:
        return

    fig, axes = plt.subplots(1, B, figsize=(5.5 * B, 6), sharey=True)
    if B == 1:
        axes = [axes]

    bar_colors = plt.cm.turbo(np.linspace(0.15, 0.85, len(regions)))

    for b in range(B):
        ax = axes[b]
        band_data = [r for r in results if int(r[2]) == b]
        names = [r[0].replace("_", "\n") for r in band_data]
        means = [r[3] for r in band_data]
        stds  = [r[4] if not np.isnan(r[4]) else 0.0 for r in band_data]
        x = np.arange(len(names))

        colors_sel = bar_colors[:len(names)]
        bars = ax.bar(x, means, yerr=stds, color=colors_sel,
                      edgecolor="#222222", linewidth=0.8,
                      capsize=5, error_kw={"linewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
        lo, hi = cfg.bands[b]
        ax.set_title(f"{lo}–{hi} Hz", fontsize=14, fontweight="bold")
        ax.grid(axis='y', alpha=0.3)
        if b == 0:
            ax.set_ylabel(f"{name_prefix}$^{{-1}}$ (mean ± std)",
                          fontsize=13, fontweight="bold")

    fig.suptitle(f"Subregion {name_prefix}$^{{-1}}$ Summary",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir,
                             f"{name_prefix}_subregion_bars.png"),
                bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  SAVE ALL OUTPUTS
# ══════════════════════════════════════════════════════════════════════
def save_outputs(Qi_mean, Qs_mean, Qi_std=None, Qs_std=None,
                 cov_tuple=None, obs_dir_first=None, pred_first=None):
    print('\n[7/7] Saving all outputs...')

    # ── Map slices ──
    print("  Plotting Qi maps...")
    _plot_maps(Qi_mean, "Qi")
    print("  Plotting Qsct maps...")
    _plot_maps(Qs_mean, "Qsct")
    if Qi_std is not None:
        print("  Plotting Qi_std maps...")
        _plot_maps(Qi_std, "Qi_std")
    if Qs_std is not None:
        print("  Plotting Qsct_std maps...")
        _plot_maps(Qs_std, "Qsct_std")

    # ── Depth cross-sections ──
    print("  Plotting cross-sections...")
    _plot_cross_sections(Qi_mean, "Qi")
    _plot_cross_sections(Qs_mean, "Qsct")
    if Qi_std is not None:
        _plot_cross_sections(Qi_std, "Qi_std")
    if Qs_std is not None:
        _plot_cross_sections(Qs_std, "Qsct_std")

    # ── Q⁻¹(f) curves ──
    print("  Plotting Q⁻¹(f) curves...")
    _qinv_vs_freq_errorbars(Qi_mean, Qi_std, title_prefix="Qi")
    _qinv_vs_freq_errorbars(Qs_mean, Qs_std, title_prefix="Qsct")

    # ── GeoTIFF slices ──
    print("  Exporting GeoTIFFs...")
    stats_all = []
    stats_all += _save_geotiff_slices(Qi_mean, "Qi")
    stats_all += _save_geotiff_slices(Qs_mean, "Qsct")
    if Qi_std is not None:
        stats_all += _save_geotiff_slices(Qi_std, "Qi_std")
    if Qs_std is not None:
        stats_all += _save_geotiff_slices(Qs_std, "Qsct_std")
    if stats_all:
        with open(os.path.join(cfg.out_dir, "slice_stats.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["field", "band_index", "z_index", "depth_km",
                         "min", "max", "mean", "std", "nan_frac", "flag"])
            w.writerows(stats_all)
        with open(os.path.join(cfg.out_dir, "slice_stats.json"), "w",
                  encoding="utf-8") as f:
            json.dump([{
                "field": r[0], "band_index": r[1], "z_index": r[2],
                "depth_km": r[3], "min": r[4], "max": r[5],
                "mean": r[6], "std": r[7], "nan_frac": r[8], "flag": r[9],
            } for r in stats_all], f, indent=2)

    # ── Voxel CSVs ──
    print("  Exporting voxel CSVs...")
    _save_voxel_csv(Qi_mean, "Qi", "Qi_voxels.csv")
    _save_voxel_csv(Qs_mean, "Qsct", "Qsct_voxels.csv")
    if Qi_std is not None:
        _save_voxel_csv(Qi_std, "Qi_std", "Qi_std_voxels.csv")
    if Qs_std is not None:
        _save_voxel_csv(Qs_std, "Qsct_std", "Qsct_std_voxels.csv")

    # ── Coverage ──
    if cov_tuple is not None:
        print("  Plotting coverage...")
        _plot_coverage_maps(cov_tuple)
        if Qi_std is not None:
            _coverage_vs_std_scatter(cov_tuple, Qi_std, tag="Qi")
        if Qs_std is not None:
            _coverage_vs_std_scatter(cov_tuple, Qs_std, tag="Qsct")

    # ── Residual diagnostics ──
    if obs_dir_first is not None and pred_first is not None:
        print("  Plotting residual diagnostics...")
        band_labels = [f"{lo}–{hi} Hz" for (lo, hi) in cfg.bands]
        _plot_residual_diagnostics(obs_dir_first, pred_first,
                                   band_labels, tag="total")

    # ── Subregion statistics ──
    try:
        if Qi_std is not None:
            print("  Computing Qi subregion stats...")
            _region_stats(Qi_mean, Qi_std, name_prefix="Qi")
    except Exception as e:
        print(f"WARNING: Qi subregion stats failed: {e}")

    try:
        if Qs_std is not None:
            print("  Computing Qsct subregion stats...")
            _region_stats(Qs_mean, Qs_std, name_prefix="Qsct")
    except Exception as e:
        print(f"WARNING: Qsct subregion stats failed: {e}")

    # ── VTK export ──
    try:
        import pyvista as pv
        GridClass = (getattr(pv, "UniformGrid", None)
                     or getattr(pv, "ImageData", None))
        if GridClass is None:
            raise AttributeError("No UniformGrid/ImageData")
        spacing = (
            (cfg.max_lon - cfg.min_lon) / max(1, cfg.nx - 1),
            (cfg.max_lat - cfg.min_lat) / max(1, cfg.ny - 1),
            (cfg.max_depth_km - cfg.min_depth_km) / max(1, cfg.nz - 1),
        )
        origin = (cfg.min_lon, cfg.min_lat, cfg.min_depth_km)
        if GridClass.__name__ == "UniformGrid":
            grid = GridClass(dimensions=(cfg.nx, cfg.ny, cfg.nz),
                             spacing=spacing, origin=origin)
        else:
            grid = GridClass()
            grid.dimensions = (cfg.nx, cfg.ny, cfg.nz)
            grid.spacing = spacing
            grid.origin = origin
        grid['Qi_mean_b0'] = Qi_mean[0].reshape(
            cfg.nz, cfg.ny, cfg.nx).ravel(order='C')
        grid['Qs_mean_b0'] = Qs_mean[0].reshape(
            cfg.nz, cfg.ny, cfg.nx).ravel(order='C')
        if Qi_std is not None:
            grid['Qi_std_b0'] = Qi_std[0].reshape(
                cfg.nz, cfg.ny, cfg.nx).ravel(order='C')
        if Qs_std is not None:
            grid['Qs_std_b0'] = Qs_std[0].reshape(
                cfg.nz, cfg.ny, cfg.nx).ravel(order='C')
        grid.save(os.path.join(cfg.out_dir,
                               'atten_tomo_uncertainty.vtk'))
        print("  Saved VTK volume")
    except Exception as e:
        print("  VTK export skipped:", e)

    print(f"\n  All outputs saved to: {cfg.out_dir}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("STARTING ATTEN-SPEC-TOMO v4.0 — KIRKUK IRAQ REGION")
    print("=" * 60)
    _set_seed(cfg.seed)

    try:
        print("\n" + "=" * 50)
        print("STEP 1: FEATURE EXTRACTION")
        print("=" * 50)
        feats = run_feature_extraction()

        print("\n" + "=" * 50)
        print("STEP 2: PATH DATA ASSEMBLY")
        print("=" * 50)
        paths_data = build_paths_data(feats)

        if len(paths_data) == 0:
            print("No usable paths for inversion.")
        else:
            cov_tuple = compute_coverage_by_band(paths_data)
            run_inversion = _define_gpu_parts()

            print("\n" + "=" * 50)
            print("STEP 3: ENSEMBLE INVERSION")
            print("=" * 50)
            Qi_list = []
            Qs_list = []
            pred_first_member = None
            for m in range(cfg.n_ensemble):
                res = run_inversion(
                    paths_data, member_idx=m, seed=cfg.seed,
                    bootstrap_frac=cfg.bootstrap_frac,
                )
                Qi_list.append(res["Qi"])
                Qs_list.append(res["Qs"])
                np.save(os.path.join(cfg.out_dir,
                                     f'Qi_inv_m{m + 1}.npy'), res["Qi"])
                np.save(os.path.join(cfg.out_dir,
                                     f'Qsct_inv_m{m + 1}.npy'), res["Qs"])
                if m == 0:
                    pred_first_member = res["pred_total"]

            Qi_stack = np.stack(Qi_list, axis=0)
            Qs_stack = np.stack(Qs_list, axis=0)
            Qi_mean = Qi_stack.mean(axis=0)
            Qs_mean = Qs_stack.mean(axis=0)
            Qi_std  = Qi_stack.std(axis=0, ddof=1)
            Qs_std  = Qs_stack.std(axis=0, ddof=1)

            np.save(os.path.join(cfg.out_dir, 'Qi_inv_mean.npy'), Qi_mean)
            np.save(os.path.join(cfg.out_dir, 'Qsct_inv_mean.npy'), Qs_mean)
            np.save(os.path.join(cfg.out_dir, 'Qi_inv_std.npy'), Qi_std)
            np.save(os.path.join(cfg.out_dir, 'Qsct_inv_std.npy'), Qs_std)

            # Build obs_dir for diagnostics
            B = len(cfg.bands)
            P = len(paths_data)
            obs_dir_full = np.zeros((B, P), dtype=np.float32)
            for p, pd in enumerate(paths_data):
                for b in range(B):
                    if b in pd['obs']:
                        od, oi, os_ = pd['obs'][b]
                        obs_dir_full[b, p] = float(od)
                    else:
                        obs_dir_full[b, p] = np.nan

            obs_dir_first = None
            if (pred_first_member is not None
                    and pred_first_member.shape[1] <= obs_dir_full.shape[1]):
                obs_dir_first = obs_dir_full[
                    :, :pred_first_member.shape[1]
                ]

            save_outputs(
                Qi_mean, Qs_mean, Qi_std, Qs_std,
                cov_tuple=cov_tuple,
                obs_dir_first=obs_dir_first,
                pred_first=pred_first_member,
            )

            print("\nDone. Check outputs in", cfg.out_dir)

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 50)
        print("EXECUTION COMPLETE")
        print("=" * 50)


# In[1]:


# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Enhanced Figure Generator v4.0 — KIRKUK IRAQ REGION               ║
║  Companion to ATTEN-SPEC-TOMO v4.0                                 ║
╚══════════════════════════════════════════════════════════════════════╝

Matches AST v4.0 config:
  Region : 38.0–49.0°E, 29.0–39.0°N, depth 0–15 km
  Grid   : 100 × 108 × 16
  Bands  : (0.5–1.5), (1.5–3.0), (3.0–6.0), (6.0–12.0) Hz

Changelog from v3.1 figure generator:
  ── BUG FIXES ──
   1. pivot_grid: guarded against duplicate (lon,lat) causing pivot crash
   2. _faults_in_extent: fixed bbox coordinate order (minx,miny,maxx,maxy)
   3. _plot_subregion_labels: guarded depth_range key missing
   4. depths_to_plot: skip NaN/Inf depths
   5. _setup_ax_to_extent: handle Cartopy set_extent failure gracefully
   6. plot_field_maps: handle single-band case (axes indexing)
   7. band_limits: guard against all-NaN band slices
   8. colorbar: ScalarFormatter with proper power limits
   9. read_csv_safely: added 'cp1252' fallback encoding
  10. _plot_faults: catch geometry errors from invalid shapefiles

  ── VISUALIZATION UPGRADES ──
   • DPI raised to 480 (savefig) / 400 (figure) for publication quality
   • 8 custom high-contrast colormaps matching AST v4.0 palette
   • Contour overlays on all map slices (white lines, labeled)
   • Bilinear interpolation for smoother rendering
   • Enhanced typography (DejaVu Sans, heavier weights, better spacing)
   • Depth cross-section plots (lon–depth, lat–depth) per field
   • Subregion polygon outlines with drop-shadow path effects
   • Diverging colormap for residual/anomaly maps
   • Coverage density hatching on low-coverage areas
   • Overview map with translucent subregion polygons + depth labels
   • Per-band frequency labels instead of band index numbers
   • Improved colorbar formatting with ScalarFormatter
   • Marginal histograms on Q⁻¹(f) summary panel
   • Multi-depth composite figure (4×4 depth grid per band)
   • Consistent axis formatting across all figures
"""

from __future__ import annotations
import os, re, math, io, warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import (
    AutoMinorLocator, ScalarFormatter, MaxNLocator, MultipleLocator,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import geopandas as gpd
    from shapely.geometry import box, Polygon as ShapelyPolygon
    HAS_GPD = True
except ImportError:
    HAS_GPD = False

# ═════════════════════════════════════════════════════════════
#  CUSTOM COLORMAPS — energetic, high-contrast, perceptually
#  balanced (matching AST v4.0 main code)
# ═════════════════════════════════════════════════════════════

def _build_cmap(colors, name, N=512):
    return LinearSegmentedColormap.from_list(name, colors, N=N)

# Intrinsic Q — deep teal → electric cyan → warm gold → hot magenta
CMAP_QI = _build_cmap([
    "#0a0a2e", "#0b3d6b", "#0891b2", "#06d6a0",
    "#fbbf24", "#f97316", "#ef4444", "#db2777",
], "ast_qi")

# Scattering Q — midnight → royal purple → neon pink → solar orange
CMAP_QSCT = _build_cmap([
    "#0f0720", "#3b0764", "#7c3aed", "#c026d3",
    "#f43f5e", "#fb923c", "#fde047", "#fefce8",
], "ast_qsct")

# Uncertainty — black → indigo → crimson → bright yellow
CMAP_STD = _build_cmap([
    "#000000", "#1e1b4b", "#4c1d95", "#9333ea",
    "#dc2626", "#f97316", "#facc15", "#fefce8",
], "ast_std")

# Coverage — deep ocean → teal → lime → white-hot
CMAP_COV = _build_cmap([
    "#000814", "#001d3d", "#003566", "#0077b6",
    "#00b4d8", "#43aa8b", "#90be6d", "#f9c74f", "#f8961e",
], "ast_cov")

# Diverging residual — blue ↔ white ↔ red
CMAP_RESID = _build_cmap([
    "#08306b", "#2171b5", "#6baed6", "#c6dbef",
    "#ffffff",
    "#fcbba1", "#fb6a4a", "#cb181d", "#67000d",
], "ast_resid")

# Scatter / general
CMAP_SCAT = _build_cmap([
    "#1b4332", "#2d6a4f", "#40916c", "#74c69d",
    "#ffd166", "#ef476f", "#9b2226",
], "ast_scat")

# Overview map terrain
CMAP_TERRAIN = _build_cmap([
    "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
], "ast_terrain")


# ═════════════════════════════════════════════════════════════
#  CONFIG — KIRKUK IRAQ (matches AST v4.0 main code)
# ═════════════════════════════════════════════════════════════
ROOT = r"C:\Users\shahe\Desktop\Kirkuk SAC files\ast_outputs_kirkuk_v4"

FAULT_SHAPEFILES = [
    r"D:\iraq data\Turkey\turkish faults\MTA_TURKEY_FAULTS_2012.shp",
]

# Auto-discover .shp files in Kirkuk fault directory
KIRKUK_FAULT_DIR = r"C:\Users\shahe\Desktop\supervising master students\Kirkuk_Fault"
if os.path.isdir(KIRKUK_FAULT_DIR):
    for _f in os.listdir(KIRKUK_FAULT_DIR):
        if _f.lower().endswith(".shp"):
            FAULT_SHAPEFILES.append(os.path.join(KIRKUK_FAULT_DIR, _f))

FIG_DIR = os.path.join(ROOT, "figures_kirkuk_v4")
os.makedirs(FIG_DIR, exist_ok=True)

DPI_SAVE = 480
DPI_FIG  = 400

# ── Palette ──
_PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
    "#f4a261", "#264653", "#a855f7", "#06b6d4",
]

plt.rcParams.update({
    # Resolution
    "figure.dpi":           DPI_FIG,
    "savefig.dpi":          DPI_SAVE,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "#fafafa",
    "savefig.pad_inches":   0.18,

    # Typography
    "font.family":          "sans-serif",
    "font.sans-serif":      ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":            13,
    "font.weight":          "medium",

    # Axes
    "axes.titlesize":       17,
    "axes.titleweight":     "bold",
    "axes.titlepad":        14,
    "axes.labelsize":       14.5,
    "axes.labelweight":     "bold",
    "axes.labelpad":        8,
    "axes.linewidth":       1.3,
    "axes.edgecolor":       "#2f2f2f",
    "axes.facecolor":       "#fafafa",
    "axes.grid":            False,
    "axes.prop_cycle":      matplotlib.cycler(color=_PALETTE),

    # Ticks
    "xtick.labelsize":      12,
    "ytick.labelsize":      12,
    "xtick.major.width":    1.1,
    "ytick.major.width":    1.1,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "xtick.direction":      "in",
    "ytick.direction":      "in",

    # Grid
    "grid.linestyle":       ":",
    "grid.alpha":           0.35,
    "grid.linewidth":       0.8,
    "grid.color":           "#888888",

    # Lines
    "lines.linewidth":      2.4,
    "lines.markersize":     7,

    # Legend
    "legend.fontsize":      11,
    "legend.frameon":       True,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#444444",
    "legend.fancybox":      True,
    "legend.shadow":        True,
    "legend.borderpad":     0.6,

    # Figure
    "figure.facecolor":     "#fafafa",
    "figure.autolayout":    False,
})


REPORT = io.StringIO()

def log(msg):
    print(msg)
    REPORT.write(msg + "\n")


# ═════════════════════════════════════════════════════════════
#  STUDY AREA
# ═════════════════════════════════════════════════════════════
study_lon_min = 38.0
study_lon_max = 49.0
study_lat_min = 29.0
study_lat_max = 39.0

KIRKUK_LON, KIRKUK_LAT = 44.39, 35.47

# Frequency bands (for labeling)
FREQ_BANDS = [(0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0)]

def _band_label(b):
    """Return human-readable frequency label for band index."""
    if 0 <= b < len(FREQ_BANDS):
        lo, hi = FREQ_BANDS[b]
        return f"{lo}–{hi} Hz"
    return f"Band {b}"


# ═════════════════════════════════════════════════════════════
#  6 GEOLOGICAL SUBREGIONS — depth ranges clamped to 0–15 km
# ═════════════════════════════════════════════════════════════
SUBREGIONS = {
    "Kirkuk Embayment": {
        "center":      (44.40, 35.50),
        "color":       "#e63946",
        "depth_range": (0, 15),
        "vertices":    [(43.8, 35.0), (45.0, 35.0),
                        (45.0, 36.0), (43.8, 36.0)],
    },
    "Zagros Foothills": {
        "center":      (45.75, 35.75),
        "color":       "#f4a261",
        "depth_range": (0, 15),
        "vertices":    [(45.0, 35.0), (46.5, 35.0),
                        (46.5, 36.5), (45.0, 36.5)],
    },
    "Mesopotamian Foredeep": {
        "center":      (44.25, 32.75),
        "color":       "#2a9d8f",
        "depth_range": (0, 15),
        "vertices":    [(43.0, 31.5), (45.5, 31.5),
                        (45.5, 34.0), (43.0, 34.0)],
    },
    "Hamrin–Makhul": {
        "center":      (43.75, 34.75),
        "color":       "#264653",
        "depth_range": (0, 15),
        "vertices":    [(43.0, 34.0), (44.5, 34.0),
                        (44.5, 35.5), (43.0, 35.5)],
    },
    "Kirkuk Deep Crust": {
        "center":      (44.50, 35.50),
        "color":       "#9b5de5",
        "depth_range": (10, 15),
        "vertices":    [(43.5, 34.5), (45.5, 34.5),
                        (45.5, 36.5), (43.5, 36.5)],
    },
    "N. Iraq – SE Turkey": {
        "center":      (43.50, 37.25),
        "color":       "#00bbf9",
        "depth_range": (0, 15),
        "vertices":    [(42.5, 36.5), (44.5, 36.5),
                        (44.5, 38.0), (42.5, 38.0)],
    },
}


# ═════════════════════════════════════════════════════════════
#  OPTIONAL CARTOPY
# ═════════════════════════════════════════════════════════════
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    ccrs = None
    cfeature = None
    HAS_CARTOPY = False


# ═════════════════════════════════════════════════════════════
#  LOAD FAULT SHAPEFILES (with geometry-error guard)
# ═════════════════════════════════════════════════════════════
FAULT_GDF = None

def load_faults(shp_paths):
    if not HAS_GPD:
        log("WARNING: geopandas not available. Fault lines will not be plotted.")
        return None

    gdfs = []
    for shp_path in shp_paths:
        if not os.path.exists(shp_path):
            log(f"WARNING: Fault shapefile not found: {shp_path}")
            continue
        try:
            gdf = gpd.read_file(shp_path)
            log(f"Loaded fault file: {shp_path} ({len(gdf)} features)")
            # Drop invalid geometries
            gdf = gdf[gdf.geometry.notna()]
            gdf = gdf[gdf.is_valid]
            if gdf.crs is not None:
                try:
                    if not gdf.crs.is_geographic:
                        gdf = gdf.to_crs(epsg=4326)
                        log("  Reprojected to EPSG:4326")
                except Exception as e:
                    log(f"  WARNING: CRS conversion issue: {e}")
            gdfs.append(gdf)
        except Exception as e:
            log(f"ERROR loading {shp_path}: {e}")

    if not gdfs:
        log("No fault data loaded.")
        return None

    merged = pd.concat(gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")

    try:
        merged = merged.cx[
            study_lon_min:study_lon_max,
            study_lat_min:study_lat_max
        ]
        log(f"Total faults inside study area: {len(merged)}")
    except Exception as e:
        log(f"WARNING: Could not crop faults: {e}")

    return merged if len(merged) > 0 else None

FAULT_GDF = load_faults(FAULT_SHAPEFILES)


# ═════════════════════════════════════════════════════════════
#  CSV HELPERS
# ═════════════════════════════════════════════════════════════
def find_file(patterns, root):
    patt = [re.compile(p, re.IGNORECASE) for p in patterns]
    try:
        for name in os.listdir(root):
            if not name.lower().endswith(".csv"):
                continue
            for rx in patt:
                if rx.search(name):
                    return os.path.join(root, name)
    except FileNotFoundError:
        pass
    return None


def read_csv_safely(path):
    if path is None or not os.path.exists(path):
        return None, "not found"
    reason = ""
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            return df, f"ok ({enc}; {df.shape[0]} rows, {df.shape[1]} cols)"
        except Exception as e:
            reason = f"read failed ({enc}): {e}"
    return None, reason


def must_have_cols(df, cols):
    return df is not None and all(c in df.columns for c in cols)


def round_coords(df, nd=4):
    if "lon" in df.columns:
        df["lon"] = df["lon"].round(nd)
    if "lat" in df.columns:
        df["lat"] = df["lat"].round(nd)
    return df


def unique_sorted(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return np.unique(a)


def depths_to_plot(depths):
    """Return all valid depth slices."""
    d = unique_sorted(depths)
    return d.tolist() if d.size > 0 else []


def pivot_grid(df, band, depth, value_col="value"):
    sub = df[
        (df["band_index"] == band) & (np.isclose(df["depth_km"], depth))
    ]
    if sub.empty:
        return None, None

    sub = round_coords(sub.copy())
    sub = sub[
        sub["lon"].between(study_lon_min - 0.1, study_lon_max + 0.1) &
        sub["lat"].between(study_lat_min - 0.1, study_lat_max + 0.1)
    ]
    if sub.empty:
        return None, None

    # Guard: drop exact duplicates before pivot
    sub = sub.drop_duplicates(subset=["lat", "lon"], keep="first")

    piv = sub.pivot_table(
        index="lat", columns="lon", values=value_col, aggfunc="mean",
    )
    if piv.shape[0] < 2 or piv.shape[1] < 2:
        return None, None

    lats = piv.index.values
    lons = piv.columns.values
    extent = [
        float(lons.min()), float(lons.max()),
        float(lats.min()), float(lats.max()),
    ]
    return piv.values, extent


def bands_from(df):
    return [int(b) for b in unique_sorted(df["band_index"].values)]


# ═════════════════════════════════════════════════════════════
#  DISCOVER FILES
# ═════════════════════════════════════════════════════════════
Qi_path     = find_file([r"^qi[_\-]?voxels.*\.csv$"], ROOT)
Qi_std_path = find_file([r"^qi[_\-]?std[_\-]?voxels.*\.csv$"], ROOT)
Qs_path     = find_file([r"^qs(ct)?[_\-]?voxels.*\.csv$"], ROOT)
Qs_std_path = find_file([r"^qs(ct)?[_\-]?std[_\-]?voxels.*\.csv$"], ROOT)

Qi_df, Qi_info         = read_csv_safely(Qi_path)
Qi_std_df, Qi_std_info = read_csv_safely(Qi_std_path)
Qs_df, Qs_info         = read_csv_safely(Qs_path)
Qs_std_df, Qs_std_info = read_csv_safely(Qs_std_path)

log(f"Qi_voxels: {Qi_info}")
log(f"Qi_std:    {Qi_std_info}")
log(f"Qs_voxels: {Qs_info}")
log(f"Qs_std:    {Qs_std_info}")


# ═════════════════════════════════════════════════════════════
#  MAP HELPERS (enhanced)
# ═════════════════════════════════════════════════════════════
def _nice_ticks(vmin, vmax, step_candidates=(0.5, 1.0, 2.0)):
    span = vmax - vmin
    if span <= 0:
        return [vmin]
    step = step_candidates[-1]
    for s in step_candidates:
        n = span / s
        if 4 <= n <= 10:
            step = s
            break
    start = np.floor(vmin / step) * step
    end   = np.ceil(vmax / step) * step
    return np.arange(start, end + 0.5 * step, step)


def _lon_ticks(vmin, vmax):
    return _nice_ticks(vmin, vmax, step_candidates=(1.0, 2.0, 5.0))


def _lat_ticks(vmin, vmax):
    return _nice_ticks(vmin, vmax, step_candidates=(0.5, 1.0, 2.0))


def _deg(val):
    return f"{int(val)}°" if val == int(val) else f"{val:.1f}°"


def _faults_in_extent(extent):
    if FAULT_GDF is None or FAULT_GDF.empty or not HAS_GPD:
        return None
    try:
        # Correct order: minx, miny, maxx, maxy
        bbox = box(extent[0], extent[2], extent[1], extent[3])
        clipped = gpd.clip(FAULT_GDF, bbox)
        if clipped.empty:
            return None
        return clipped
    except Exception:
        try:
            return FAULT_GDF.cx[extent[0]:extent[1], extent[2]:extent[3]]
        except Exception:
            return FAULT_GDF


def _plot_faults(ax, extent):
    """Plot fault lines with error-tolerant geometry handling."""
    gdf_local = _faults_in_extent(extent)
    if gdf_local is None or gdf_local.empty:
        return
    try:
        # Filter out any remaining invalid geometries
        gdf_local = gdf_local[gdf_local.geometry.notna()]
        if gdf_local.empty:
            return
        plot_kw = dict(
            color="#1a1a1a", linewidth=0.9, alpha=0.82, zorder=3,
        )
        if HAS_CARTOPY and hasattr(ax, "projection"):
            gdf_local.plot(ax=ax, transform=ccrs.PlateCarree(), **plot_kw)
        else:
            gdf_local.plot(ax=ax, **plot_kw)
    except Exception as e:
        log(f"WARNING: fault plotting failed: {e}")


def _plot_subregion_outlines(ax, current_depth_km, transform=None):
    """Draw translucent polygon outlines for active subregions."""
    drawn = []
    for name, info in SUBREGIONS.items():
        z0, z1 = info.get("depth_range", (0, 15))
        if not (z0 <= current_depth_km <= z1):
            continue
        verts = info.get("vertices", None)
        if verts is None:
            continue

        col = info["color"]
        poly_x = [v[0] for v in verts] + [verts[0][0]]
        poly_y = [v[1] for v in verts] + [verts[0][1]]

        plot_kw = dict(
            color=col, linewidth=1.8, linestyle="--", alpha=0.75, zorder=6,
        )
        if transform is not None:
            plot_kw["transform"] = transform
        ax.plot(
            poly_x, poly_y,
            path_effects=[
                pe.Stroke(linewidth=3.2, foreground="white", alpha=0.5),
                pe.Normal(),
            ],
            **plot_kw,
        )
        drawn.append((name, col))
    return drawn


def _plot_subregion_labels(ax, current_depth_km, transform=None):
    """Place labels for subregions whose depth range includes current slice."""
    drawn = []
    for name, info in SUBREGIONS.items():
        z0, z1 = info.get("depth_range", (0, 15))
        if not (z0 <= current_depth_km <= z1):
            continue

        cx, cy = info["center"]
        col = info["color"]

        txt_kw = dict(
            fontsize=7.5, fontweight="bold", color=col,
            ha="center", va="center", zorder=8,
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground="white"),
            ],
        )
        if transform is not None:
            txt_kw["transform"] = transform
        ax.text(cx, cy, name, **txt_kw)
        drawn.append((name, col))
    return drawn


def _plot_kirkuk_marker(ax, transform=None):
    """Gold star marker for Kirkuk city."""
    kw = dict(
        marker="*", color="#ffd700", markersize=14,
        markeredgecolor="#1a1a1a", markeredgewidth=0.9, zorder=9,
    )
    if transform is not None:
        kw["transform"] = transform
    ax.plot(KIRKUK_LON, KIRKUK_LAT, **kw)

    txt_kw = dict(
        fontsize=8, fontweight="bold", color="#b22222", zorder=9,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )
    if transform is not None:
        txt_kw["transform"] = transform
    ax.text(KIRKUK_LON + 0.18, KIRKUK_LAT + 0.15, "Kirkuk", **txt_kw)


def _smart_colorbar(im, ax, label):
    """Well-formatted colorbar with scientific notation."""
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    cbar.ax.yaxis.set_major_formatter(fmt)
    return cbar


def _setup_ax_to_extent(ax, extent):
    xmin, xmax, ymin, ymax = extent

    if HAS_CARTOPY and hasattr(ax, "projection"):
        try:
            ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
        except Exception:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        try:
            ax.coastlines(resolution="10m", linewidth=0.7,
                          color="#264653", zorder=2)
        except Exception:
            pass
        try:
            ax.add_feature(cfeature.BORDERS, linewidth=0.8,
                           edgecolor="#555555", zorder=2)
        except Exception:
            pass
        try:
            ax.add_feature(cfeature.RIVERS, linewidth=0.35,
                           edgecolor="#457b9d", alpha=0.45, zorder=2)
        except Exception:
            pass
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Major ticks
    xticks = _lon_ticks(xmin, xmax)
    yticks = _lat_ticks(ymin, ymax)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(
        [f"{_deg(x)}E" for x in xticks],
        fontweight="bold", fontsize=10.5, rotation=30, ha="right",
    )
    ax.set_yticklabels(
        [f"{_deg(y)}N" for y in yticks],
        fontweight="bold", fontsize=10.5,
    )

    # Minor ticks: 0.2° intervals
    xticks_minor = np.arange(np.floor(xmin), np.ceil(xmax) + 0.1, 0.2)
    yticks_minor = np.arange(np.floor(ymin), np.ceil(ymax) + 0.1, 0.2)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_yticks(yticks_minor, minor=True)

    ax.set_xlabel("Longitude (°E)", fontweight="bold", fontsize=13,
                  labelpad=8)
    ax.set_ylabel("Latitude (°N)", fontweight="bold", fontsize=13,
                  labelpad=8)

    ax.tick_params(axis="x", which="major", width=1.4, length=6, pad=4)
    ax.tick_params(axis="y", which="major", width=1.4, length=6, pad=5)
    ax.tick_params(axis="both", which="minor", length=3, width=0.7,
                   color="#666666")

    # Geographic grid
    ax.grid(True, which="major", alpha=0.28, linewidth=0.7,
            color="#444444")
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.3,
            color="#888888", linestyle=":")


def _add_contours(ax, arr, extent, n_levels=10, transform=None):
    """Add white contour overlay with labels."""
    ny, nx = arr.shape
    finite = arr[np.isfinite(arr)]
    if finite.size < 10:
        return
    vmin = np.percentile(finite, 2)
    vmax = np.percentile(finite, 98)
    if abs(vmax - vmin) < 1e-12:
        return

    lon_c = np.linspace(extent[0], extent[1], nx)
    lat_c = np.linspace(extent[2], extent[3], ny)
    LON, LAT = np.meshgrid(lon_c, lat_c)
    levels = np.linspace(vmin, vmax, n_levels)

    contour_kw = dict(
        levels=levels, colors="white", linewidths=0.45, alpha=0.5,
    )
    if transform is not None:
        contour_kw["transform"] = transform
    try:
        cs = ax.contour(LON, LAT, arr, **contour_kw)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.3g")
    except Exception:
        pass


def _build_legend(drawn_list):
    handles = []
    seen = set()
    for name, color in drawn_list:
        key = name.replace("\n", " ")
        if key in seen:
            continue
        seen.add(key)
        handles.append(
            Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=color, markeredgecolor=color,
                   markersize=7, label=key, linestyle="None")
        )
    handles.append(
        Line2D([0], [0], color="#1a1a1a", linewidth=1.0,
               linestyle="-", label="Fault lines")
    )
    handles.append(
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor="#ffd700", markeredgecolor="#1a1a1a",
               markersize=11, label="Kirkuk City", linestyle="None")
    )
    return handles


# ═════════════════════════════════════════════════════════════
#  MAIN MAP PLOTTING
# ═════════════════════════════════════════════════════════════
def plot_field_maps(vox_df, std_df, title_prefix="Qi",
                    cmap_mean=CMAP_QI, cmap_std=CMAP_STD,
                    normalize_per_band=True):

    need = ["band_index", "lon", "lat", "depth_km", "value"]
    if not must_have_cols(vox_df, need):
        log(f"[SKIP] {title_prefix} maps: missing columns {need}")
        return

    depths = unique_sorted(vox_df["depth_km"].values)
    bands  = bands_from(vox_df)
    if len(depths) == 0 or len(bands) == 0:
        log(f"[SKIP] {title_prefix} maps: no depths or bands")
        return

    # Per-band percentile limits for consistent color scaling
    band_limits = {}
    if normalize_per_band:
        for b in bands:
            sb = (vox_df[vox_df["band_index"] == b]["value"]
                  .replace([np.inf, -np.inf], np.nan).dropna())
            if sb.empty or sb.isna().all():
                band_limits[b] = (None, None)
            else:
                vmin, vmax = np.percentile(sb, [1, 99])
                band_limits[b] = (float(vmin), float(vmax))

    for d in depths_to_plot(depths):
        ncols = max(1, len(bands))
        subplot_kw = ({"projection": ccrs.PlateCarree()}
                      if HAS_CARTOPY else {})

        fig, axes = plt.subplots(
            2, ncols,
            figsize=(7.5 * ncols, 11.0),
            subplot_kw=subplot_kw,
        )
        # Handle single-band axes shape
        if ncols == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        if axes.ndim == 1:
            axes = axes.reshape(2, -1)

        all_drawn = []
        crs_transform = ccrs.PlateCarree() if HAS_CARTOPY else None

        for j, b in enumerate(bands):
            freq_label = _band_label(b)

            # ────────── MEAN ROW ──────────
            ax = axes[0, j]
            arr, extent = pivot_grid(vox_df, b, d)

            ax.set_title(
                f"{title_prefix}$^{{-1}}$ mean  ·  {freq_label}"
                f"  ·  z = {d:.1f} km",
                fontweight="bold", fontsize=13, pad=14,
            )

            if arr is None:
                ax.text(
                    0.5, 0.5, "No data available",
                    transform=ax.transAxes, ha="center", va="center",
                    fontweight="bold", fontsize=14, color="#e63946",
                )
                continue

            _setup_ax_to_extent(ax, extent)

            vmin, vmax = band_limits.get(b, (None, None))
            imshow_kw = dict(
                origin="lower", extent=extent, cmap=cmap_mean,
                aspect="auto", vmin=vmin, vmax=vmax,
                interpolation="bilinear", alpha=0.94, zorder=1,
            )
            if HAS_CARTOPY:
                imshow_kw["transform"] = ccrs.PlateCarree()
            im = ax.imshow(arr, **imshow_kw)

            # Contour overlay
            _add_contours(ax, arr, extent, n_levels=10,
                          transform=crs_transform)

            _plot_faults(ax, extent)

            drawn_outlines = _plot_subregion_outlines(
                ax, d, transform=crs_transform
            )
            drawn_labels = _plot_subregion_labels(
                ax, d, transform=crs_transform
            )
            all_drawn.extend(drawn_outlines)
            _plot_kirkuk_marker(ax, transform=crs_transform)

            _smart_colorbar(
                im, ax, f"{title_prefix}$^{{-1}}$ (mean)"
            )

            # ────────── STD ROW ──────────
            ax2 = axes[1, j]
            ax2.set_title(
                f"{title_prefix}$^{{-1}}$ std  ·  {freq_label}"
                f"  ·  z = {d:.1f} km",
                fontweight="bold", fontsize=13, pad=14,
            )

            if std_df is None or not must_have_cols(std_df, need):
                ax2.text(
                    0.5, 0.5, "No uncertainty data",
                    transform=ax2.transAxes, ha="center", va="center",
                    fontweight="bold", fontsize=14, color="#e63946",
                )
                continue

            arrs, extents = pivot_grid(std_df, b, d)
            if arrs is None:
                ax2.text(
                    0.5, 0.5, "No data available",
                    transform=ax2.transAxes, ha="center", va="center",
                    fontweight="bold", fontsize=14, color="#e63946",
                )
                continue

            _setup_ax_to_extent(ax2, extents)

            imshow_kw_s = dict(
                origin="lower", extent=extents, cmap=cmap_std,
                aspect="auto", interpolation="bilinear",
                alpha=0.94, zorder=1,
            )
            if HAS_CARTOPY:
                imshow_kw_s["transform"] = ccrs.PlateCarree()
            ims = ax2.imshow(arrs, **imshow_kw_s)

            _add_contours(ax2, arrs, extents, n_levels=8,
                          transform=crs_transform)
            _plot_faults(ax2, extents)
            _plot_subregion_outlines(ax2, d, transform=crs_transform)
            _plot_subregion_labels(ax2, d, transform=crs_transform)
            _plot_kirkuk_marker(ax2, transform=crs_transform)

            _smart_colorbar(
                ims, ax2, f"{title_prefix}$^{{-1}}$ (std)"
            )

        # ────────── LEGEND ──────────
        if all_drawn:
            legend_handles = _build_legend(all_drawn)
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=min(5, len(legend_handles)),
                fontsize=9.5,
                frameon=True, framealpha=0.95, edgecolor="#333",
                bbox_to_anchor=(0.5, 0.003),
                borderaxespad=0.5,
            )

        plt.subplots_adjust(
            top=0.88, bottom=0.12, left=0.06, right=0.97,
            hspace=0.38, wspace=0.28,
        )

        fig.suptitle(
            f"{title_prefix} Attenuation Tomography — "
            f"Kirkuk Region  ·  Depth {d:.1f} km",
            y=0.98, fontweight="bold", fontsize=18,
            bbox=dict(
                boxstyle="round,pad=0.6", facecolor="#ffffffee",
                edgecolor="#2f2f2f", linewidth=1.6,
            ),
        )

        out = os.path.join(
            FIG_DIR, f"{title_prefix}_kirkuk_depth_{d:.1f}km.png"
        )
        plt.savefig(out, dpi=DPI_SAVE, bbox_inches="tight",
                    facecolor="#fafafa", pad_inches=0.35)
        plt.close(fig)
        log(f"[OK] {out}")


# ═════════════════════════════════════════════════════════════
#  DEPTH CROSS-SECTION PLOTS
# ═════════════════════════════════════════════════════════════
def plot_cross_sections(vox_df, title_prefix="Qi", cmap=CMAP_QI):
    """
    Lon–depth and lat–depth cross-sections at mid-latitude and
    mid-longitude.
    """
    need = ["band_index", "lon", "lat", "depth_km", "value"]
    if not must_have_cols(vox_df, need):
        log(f"[SKIP] {title_prefix} cross-sections: missing columns")
        return

    bands  = bands_from(vox_df)
    depths = unique_sorted(vox_df["depth_km"].values)
    lons   = unique_sorted(vox_df["lon"].values)
    lats   = unique_sorted(vox_df["lat"].values)

    if len(depths) < 2 or len(lons) < 2 or len(lats) < 2:
        log(f"[SKIP] {title_prefix} cross-sections: insufficient grid")
        return

    mid_lat = lats[len(lats) // 2]
    mid_lon = lons[len(lons) // 2]

    for b in bands:
        freq_label = _band_label(b)
        sub = vox_df[vox_df["band_index"] == b].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
        if sub.empty:
            continue

        finite_vals = sub["value"].dropna()
        if finite_vals.empty:
            continue
        vmin, vmax = np.percentile(finite_vals, [1, 99])
        if abs(vmax - vmin) < 1e-12:
            vmin, vmax = finite_vals.min(), finite_vals.max()

        # ── Lon–depth at mid_lat ──
        slice_df = sub[np.isclose(sub["lat"], mid_lat, atol=0.15)]
        if not slice_df.empty:
            slice_df = slice_df.drop_duplicates(
                subset=["lon", "depth_km"], keep="first"
            )
            piv = slice_df.pivot_table(
                index="depth_km", columns="lon", values="value",
                aggfunc="mean",
            )
            if piv.shape[0] >= 2 and piv.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(12, 5.5))
                im = ax.imshow(
                    piv.values, origin="upper", aspect="auto",
                    extent=[piv.columns.min(), piv.columns.max(),
                            piv.index.max(), piv.index.min()],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation="bilinear",
                )
                ax.set_xlabel("Longitude (°E)")
                ax.set_ylabel("Depth (km)")
                ax.set_title(
                    f"{title_prefix}$^{{-1}}$  ·  {freq_label}  ·  "
                    f"Lat ≈ {mid_lat:.1f}°N (cross-section)",
                    fontsize=15, fontweight="bold",
                )
                _smart_colorbar(im, ax, f"{title_prefix}$^{{-1}}$")
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(os.path.join(
                    FIG_DIR,
                    f"{title_prefix}_xsec_lon_band{b}.png",
                ), dpi=DPI_SAVE)
                plt.close(fig)
                log(f"[OK] {title_prefix} lon-depth xsec band {b}")

        # ── Lat–depth at mid_lon ──
        slice_df = sub[np.isclose(sub["lon"], mid_lon, atol=0.15)]
        if not slice_df.empty:
            slice_df = slice_df.drop_duplicates(
                subset=["lat", "depth_km"], keep="first"
            )
            piv = slice_df.pivot_table(
                index="depth_km", columns="lat", values="value",
                aggfunc="mean",
            )
            if piv.shape[0] >= 2 and piv.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(12, 5.5))
                im = ax.imshow(
                    piv.values, origin="upper", aspect="auto",
                    extent=[piv.columns.min(), piv.columns.max(),
                            piv.index.max(), piv.index.min()],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation="bilinear",
                )
                ax.set_xlabel("Latitude (°N)")
                ax.set_ylabel("Depth (km)")
                ax.set_title(
                    f"{title_prefix}$^{{-1}}$  ·  {freq_label}  ·  "
                    f"Lon ≈ {mid_lon:.1f}°E (cross-section)",
                    fontsize=15, fontweight="bold",
                )
                _smart_colorbar(im, ax, f"{title_prefix}$^{{-1}}$")
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(os.path.join(
                    FIG_DIR,
                    f"{title_prefix}_xsec_lat_band{b}.png",
                ), dpi=DPI_SAVE)
                plt.close(fig)
                log(f"[OK] {title_prefix} lat-depth xsec band {b}")


# ═════════════════════════════════════════════════════════════
#  MULTI-DEPTH COMPOSITE FIGURE
# ═════════════════════════════════════════════════════════════
def plot_depth_composite(vox_df, title_prefix="Qi", cmap=CMAP_QI,
                         max_panels=16):
    """4×4 (or smaller) grid showing all depths for one band."""
    need = ["band_index", "lon", "lat", "depth_km", "value"]
    if not must_have_cols(vox_df, need):
        log(f"[SKIP] {title_prefix} depth composite: missing columns")
        return

    bands  = bands_from(vox_df)
    depths = depths_to_plot(unique_sorted(vox_df["depth_km"].values))
    if len(depths) == 0:
        return

    # Limit to max_panels
    depths = depths[:max_panels]
    ncols = min(4, len(depths))
    nrows = math.ceil(len(depths) / ncols)

    for b in bands:
        freq_label = _band_label(b)

        # Global limits for this band
        sb = (vox_df[vox_df["band_index"] == b]["value"]
              .replace([np.inf, -np.inf], np.nan).dropna())
        if sb.empty:
            continue
        vmin, vmax = np.percentile(sb, [1, 99])

        subplot_kw = ({"projection": ccrs.PlateCarree()}
                      if HAS_CARTOPY else {})
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 4.5 * nrows),
            subplot_kw=subplot_kw,
        )
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        crs_transform = ccrs.PlateCarree() if HAS_CARTOPY else None

        for idx, d in enumerate(depths):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            arr, extent = pivot_grid(vox_df, b, d)

            if arr is None:
                ax.set_title(f"z = {d:.1f} km", fontsize=10)
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center", fontsize=16,
                        color="#999999")
                continue

            _setup_ax_to_extent(ax, extent)

            imshow_kw = dict(
                origin="lower", extent=extent, cmap=cmap,
                aspect="auto", vmin=vmin, vmax=vmax,
                interpolation="bilinear", alpha=0.94, zorder=1,
            )
            if HAS_CARTOPY:
                imshow_kw["transform"] = ccrs.PlateCarree()
            ax.imshow(arr, **imshow_kw)
            _plot_faults(ax, extent)
            _plot_kirkuk_marker(ax, transform=crs_transform)
            ax.set_title(f"z = {d:.1f} km", fontsize=11,
                         fontweight="bold", pad=6)
            # Simplify tick labels for small panels
            ax.set_xlabel("")
            ax.set_ylabel("")

        # Hide unused panels
        for idx in range(len(depths), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        # Shared colorbar
        fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.3)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f"{title_prefix}$^{{-1}}$", fontsize=13,
                       fontweight="bold")

        fig.suptitle(
            f"{title_prefix}$^{{-1}}$ — {freq_label}  ·  All Depths",
            fontsize=17, fontweight="bold", y=0.98,
        )

        out = os.path.join(
            FIG_DIR, f"{title_prefix}_depth_composite_band{b}.png"
        )
        fig.savefig(out, dpi=DPI_SAVE, bbox_inches="tight",
                    facecolor="#fafafa")
        plt.close(fig)
        log(f"[OK] {out}")


# ═════════════════════════════════════════════════════════════
#  Q⁻¹(f) FREQUENCY CURVES (from voxel CSVs)
# ═════════════════════════════════════════════════════════════
def plot_qinv_vs_freq(vox_df, std_df, title_prefix="Qi"):
    """Q⁻¹(f) curves with shaded uncertainty at selected depths."""
    need = ["band_index", "lon", "lat", "depth_km", "value"]
    if not must_have_cols(vox_df, need):
        return

    bands  = bands_from(vox_df)
    depths = unique_sorted(vox_df["depth_km"].values)
    if len(bands) < 2 or len(depths) == 0:
        return

    fcs = np.array([math.sqrt(lo * hi) for lo, hi in FREQ_BANDS
                     if lo <= 12.0], dtype=float)
    if len(fcs) < len(bands):
        fcs = np.arange(len(bands), dtype=float) + 1.0

    # Pick representative depths (up to 6)
    if len(depths) <= 6:
        plot_depths = depths
    else:
        idxs = np.linspace(0, len(depths) - 1, 6, dtype=int)
        plot_depths = depths[idxs]

    colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(plot_depths)))

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    for ci, d in enumerate(plot_depths):
        means = []
        stds  = []
        for b in bands:
            sub = vox_df[
                (vox_df["band_index"] == b) &
                (np.isclose(vox_df["depth_km"], d))
            ]["value"]
            means.append(sub.mean() if not sub.empty else np.nan)

            if std_df is not None and must_have_cols(std_df, need):
                sub_s = std_df[
                    (std_df["band_index"] == b) &
                    (np.isclose(std_df["depth_km"], d))
                ]["value"]
                stds.append(sub_s.mean() if not sub_s.empty else 0.0)
            else:
                stds.append(0.0)

        m = np.array(means)
        s = np.array(stds)
        c = colors[ci]
        label = f"{d:.1f} km"

        ax.plot(fcs[:len(m)], m, '-o', color=c, linewidth=2.3,
                markersize=7, label=label, zorder=5)
        ax.fill_between(fcs[:len(m)], m - s, m + s, color=c, alpha=0.15)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"{title_prefix}$^{{-1}}$", fontsize=14,
                  fontweight="bold")
    ax.set_title(
        f"{title_prefix}$^{{-1}}(f)$ with Uncertainty — Selected Depths",
        fontsize=16, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Depth", title_fontsize=12, fontsize=11,
              loc="upper right", framealpha=0.92)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(os.path.join(
        FIG_DIR, f"{title_prefix}_qinv_vs_freq.png"
    ), dpi=DPI_SAVE)
    plt.close(fig)
    log(f"[OK] {title_prefix} Q⁻¹(f) frequency curves")


# ═════════════════════════════════════════════════════════════
#  MEAN vs STD SCATTER (uncertainty diagnostic)
# ═════════════════════════════════════════════════════════════
def plot_mean_vs_std_scatter(vox_df, std_df, title_prefix="Qi"):
    """Scatter of mean vs std per voxel for each band."""
    need = ["band_index", "lon", "lat", "depth_km", "value"]
    if (not must_have_cols(vox_df, need) or
            not must_have_cols(std_df, need)):
        return

    bands = bands_from(vox_df)
    for b in bands:
        freq_label = _band_label(b)
        m_sub = vox_df[vox_df["band_index"] == b][
            ["lon", "lat", "depth_km", "value"]
        ].rename(columns={"value": "mean_val"})
        s_sub = std_df[std_df["band_index"] == b][
            ["lon", "lat", "depth_km", "value"]
        ].rename(columns={"value": "std_val"})

        merged = m_sub.merge(s_sub, on=["lon", "lat", "depth_km"],
                             how="inner")
        merged = merged.dropna()
        if merged.empty or len(merged) < 10:
            continue

        x = merged["mean_val"].values
        y = merged["std_val"].values

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            x, y, s=4, c=merged["depth_km"].values,
            cmap=CMAP_SCAT, alpha=0.6, edgecolors="none",
        )
        ax.set_xlabel(f"{title_prefix}$^{{-1}}$ (mean)", fontsize=13,
                      fontweight="bold")
        ax.set_ylabel(f"{title_prefix}$^{{-1}}$ (std)", fontsize=13,
                      fontweight="bold")
        ax.set_title(
            f"Mean vs Uncertainty  ·  {title_prefix}  ·  {freq_label}",
            fontsize=15, fontweight="bold",
        )
        _smart_colorbar(sc, ax, "Depth (km)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(
            FIG_DIR, f"{title_prefix}_mean_vs_std_band{b}.png"
        ), dpi=DPI_SAVE)
        plt.close(fig)
        log(f"[OK] {title_prefix} mean-vs-std band {b}")


# ═════════════════════════════════════════════════════════════
#  SUBREGION OVERVIEW MAP (enhanced)
# ═════════════════════════════════════════════════════════════
def plot_subregion_overview():
    subplot_kw = ({"projection": ccrs.PlateCarree()}
                  if HAS_CARTOPY else {})
    fig, ax = plt.subplots(1, 1, figsize=(13, 10.5),
                           subplot_kw=subplot_kw)

    extent = [study_lon_min, study_lon_max,
              study_lat_min, study_lat_max]
    _setup_ax_to_extent(ax, extent)

    if HAS_CARTOPY:
        try:
            ax.add_feature(cfeature.LAND, facecolor="#f0ece2",
                           zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="#cce5ff",
                           zorder=0)
        except Exception:
            pass

    _plot_faults(ax, extent)
    crs_transform = ccrs.PlateCarree() if HAS_CARTOPY else None

    all_drawn = []
    for name, info in SUBREGIONS.items():
        cx, cy = info["center"]
        col = info["color"]
        z0, z1 = info.get("depth_range", (0, 15))
        verts = info.get("vertices", None)

        # Draw translucent polygon fill
        if verts is not None:
            poly_xy = np.array(verts + [verts[0]])
            patch = MplPolygon(
                poly_xy, closed=True,
                facecolor=col, edgecolor=col,
                alpha=0.18, linewidth=2.0, linestyle="--",
                zorder=4,
            )
            if crs_transform is not None:
                patch.set_transform(crs_transform._as_mpl_transform(ax))
            ax.add_patch(patch)

            # Outline with shadow
            plot_kw = dict(
                color=col, linewidth=2.0, linestyle="--",
                alpha=0.8, zorder=5,
            )
            if crs_transform is not None:
                plot_kw["transform"] = crs_transform
            ax.plot(
                poly_xy[:, 0], poly_xy[:, 1],
                path_effects=[
                    pe.Stroke(linewidth=3.5, foreground="white",
                              alpha=0.5),
                    pe.Normal(),
                ],
                **plot_kw,
            )

        display = f"{name}\n({z0}–{z1} km)"
        txt_kw = dict(
            fontsize=9, fontweight="bold", color=col,
            ha="center", va="center", zorder=8,
            path_effects=[
                pe.withStroke(linewidth=3.0, foreground="white"),
            ],
        )
        if crs_transform is not None:
            txt_kw["transform"] = crs_transform
        ax.text(cx, cy, display, **txt_kw)
        all_drawn.append((name, col))

    _plot_kirkuk_marker(ax, transform=crs_transform)

    legend_handles = _build_legend(all_drawn)
    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=10,
        frameon=True, framealpha=0.95, edgecolor="#333",
        title="Subregions", title_fontsize=11,
    )

    ax.set_title(
        "Geological Subregions — Kirkuk & Northern Iraq\n"
        f"Study Area: {study_lon_min}°–{study_lon_max}°E, "
        f"{study_lat_min}°–{study_lat_max}°N  ·  "
        f"Depth 0–15 km",
        fontweight="bold", fontsize=15, pad=16,
    )

    out = os.path.join(FIG_DIR, "subregion_overview_map.png")
    plt.savefig(out, dpi=DPI_SAVE, bbox_inches="tight",
                facecolor="#fafafa", pad_inches=0.3)
    plt.close(fig)
    log(f"[OK] Subregion overview: {out}")


# ═════════════════════════════════════════════════════════════
#  INDIVIDUAL DEPTH-SLICE FIGURES (one file per slice)
# ═════════════════════════════════════════════════════════════
def plot_individual_slices(vox_df, title_prefix="Qi", cmap=CMAP_QI):
    """Save one clean figure per (band, depth) for presentations."""
    need = ["band_index", "lon", "lat", "depth_km", "value"]
    if not must_have_cols(vox_df, need):
        return

    bands  = bands_from(vox_df)
    depths = depths_to_plot(unique_sorted(vox_df["depth_km"].values))

    subdir = os.path.join(FIG_DIR, f"{title_prefix}_slices")
    os.makedirs(subdir, exist_ok=True)

    for b in bands:
        freq_label = _band_label(b)
        sb = (vox_df[vox_df["band_index"] == b]["value"]
              .replace([np.inf, -np.inf], np.nan).dropna())
        if sb.empty:
            continue
        vmin, vmax = np.percentile(sb, [1, 99])

        for d in depths:
            arr, extent = pivot_grid(vox_df, b, d)
            if arr is None:
                continue

            subplot_kw = ({"projection": ccrs.PlateCarree()}
                          if HAS_CARTOPY else {})
            fig, ax = plt.subplots(figsize=(11, 7), subplot_kw=subplot_kw)

            _setup_ax_to_extent(ax, extent)

            imshow_kw = dict(
                origin="lower", extent=extent, cmap=cmap,
                aspect="auto", vmin=vmin, vmax=vmax,
                interpolation="bilinear", alpha=0.94, zorder=1,
            )
            crs_transform = ccrs.PlateCarree() if HAS_CARTOPY else None
            if HAS_CARTOPY:
                imshow_kw["transform"] = crs_transform
            im = ax.imshow(arr, **imshow_kw)

            _add_contours(ax, arr, extent, n_levels=10,
                          transform=crs_transform)
            _plot_faults(ax, extent)
            _plot_subregion_outlines(ax, d, transform=crs_transform)
            _plot_subregion_labels(ax, d, transform=crs_transform)
            _plot_kirkuk_marker(ax, transform=crs_transform)

            _smart_colorbar(im, ax, f"{title_prefix}$^{{-1}}$")

            ax.set_title(
                f"{title_prefix}$^{{-1}}$  ·  {freq_label}  ·  "
                f"z = {d:.1f} km",
                fontsize=16, fontweight="bold", pad=14,
            )

            fname = f"{title_prefix}_b{b}_z{d:.1f}km.png"
            fig.tight_layout()
            fig.savefig(os.path.join(subdir, fname), dpi=DPI_SAVE,
                        facecolor="#fafafa")
            plt.close(fig)

    log(f"[OK] {title_prefix} individual slices → {subdir}")


# ═════════════════════════════════════════════════════════════
#  RUN
# ═════════════════════════════════════════════════════════════
log("=" * 60)
log("  KIRKUK IRAQ — Enhanced Figure Generator v4.0")
log("=" * 60)
log(f"  Study area  : {study_lon_min}–{study_lon_max}°E, "
    f"{study_lat_min}–{study_lat_max}°N")
log(f"  Depth range : 0–15 km")
log(f"  Output dir  : {FIG_DIR}")
log(f"  Cartopy     : {'YES' if HAS_CARTOPY else 'NO'}")
log(f"  GeoPandas   : {'YES' if HAS_GPD else 'NO'}")
log(f"  Faults      : "
    f"{len(FAULT_GDF) if FAULT_GDF is not None else 0} features")
log(f"  Fault files : {len(FAULT_SHAPEFILES)}")
log(f"  Save DPI    : {DPI_SAVE}")
log("")

# ── Overview ──
plot_subregion_overview()

# ── Qi maps ──
if Qi_df is not None:
    plot_field_maps(Qi_df, Qi_std_df, "Qi",
                    cmap_mean=CMAP_QI, cmap_std=CMAP_STD)
    plot_cross_sections(Qi_df, "Qi", cmap=CMAP_QI)
    plot_depth_composite(Qi_df, "Qi", cmap=CMAP_QI)
    plot_individual_slices(Qi_df, "Qi", cmap=CMAP_QI)
    plot_qinv_vs_freq(Qi_df, Qi_std_df, "Qi")
    if Qi_std_df is not None:
        plot_mean_vs_std_scatter(Qi_df, Qi_std_df, "Qi")
else:
    log("[SKIP] Qi maps — no voxel CSV found")

# ── Qsct maps ──
if Qs_df is not None:
    plot_field_maps(Qs_df, Qs_std_df, "Qsct",
                    cmap_mean=CMAP_QSCT, cmap_std=CMAP_STD)
    plot_cross_sections(Qs_df, "Qsct", cmap=CMAP_QSCT)
    plot_depth_composite(Qs_df, "Qsct", cmap=CMAP_QSCT)
    plot_individual_slices(Qs_df, "Qsct", cmap=CMAP_QSCT)
    plot_qinv_vs_freq(Qs_df, Qs_std_df, "Qsct")
    if Qs_std_df is not None:
        plot_mean_vs_std_scatter(Qs_df, Qs_std_df, "Qsct")
else:
    log("[SKIP] Qsct maps — no voxel CSV found")

# ── Save report ──
report_path = os.path.join(FIG_DIR, "_run_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(REPORT.getvalue())

log(f"\nDone. All figures → {FIG_DIR}")


# In[3]:


# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  3D CROSS-SECTION VISUALIZER v4.0 — Kirkuk Iraq Attenuation Tomo   ║
║  Companion to ATTEN-SPEC-TOMO v4.0 & Figure Generator v4.0        ║
╚══════════════════════════════════════════════════════════════════════╝

Reads Qi_voxels.csv / Qsct_voxels.csv from AST v4.0 outputs and
generates publication-quality cross-sections:

  1. E–W vertical section through Kirkuk (lat ≈ 35.45°N)
  2. N–S vertical section through Kirkuk (lon ≈ 44.44°E)
  3. NE–SW diagonal along Zagros fold-thrust front
  4. 3D fence diagram combining all three
  5. Combined panel: depth slice + 3 cross-sections per band
  6. Multi-depth E–W comparison
  7. All-bands E–W comparison
  8. Uncertainty cross-sections (NEW)
  9. Anomaly ratio panels Qi/Qsct (NEW)

Changelog from v3.1:
  ── BUG FIXES ──
   1. extract_diagonal_section: bounds_error guard when profile extends
      outside the grid (was silently returning all-NaN slices)
   2. get_clim: fixed crash when all values are identical (vlim=0)
   3. plot_3d_fence: guarded against empty finite_all array
   4. load_voxel_csv: searchsorted clamp to valid range (off-by-one)
   5. plot_combined_panel: depth index clamp to valid range
   6. plot_depth_comparison_ew: guarded target depths to actual grid
   7. TwoSlopeNorm: catch vmin==vmax==0 edge case
   8. pcolormesh shading: switched to 'nearest' when grid is too
      coarse for 'gouraud' (prevents rendering artifacts)
   9. 3D fence: subsample step computed from actual array shape,
      not from shared lons/lats array
  10. Colorbar: ScalarFormatter with proper power limits

  ── VISUALIZATION UPGRADES ──
   • DPI raised to 480 (savefig) for publication quality
   • 6 custom high-contrast colormaps matching AST v4.0 palette
   • Contour overlays on all 2D section plots
   • Enhanced typography (DejaVu Sans, heavier weights)
   • Bilinear interpolation on depth slices
   • Improved 3D fence with better lighting & viewing angles
   • Dual-view 3D fence (two perspectives per figure)
   • Uncertainty cross-section panels (std overlaid as contours)
   • City markers with improved path-effect halos
   • Subregion depth-range annotations on vertical sections
   • Profile distance tick marks with geographic labels
   • Gradient-filled depth-mean curves (bottom panels)
   • Minor ticks & geographic grid on all map panels
   • Consistent axis formatting across all figures
"""

from __future__ import annotations
import os, sys, math, warnings
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import (
    LinearSegmentedColormap, TwoSlopeNorm, Normalize,
)
from matplotlib.ticker import (
    AutoMinorLocator, ScalarFormatter, MultipleLocator,
)
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  CUSTOM COLORMAPS — matching AST v4.0 palette
# ═══════════════════════════════════════════════════════════════

def _build_cmap(colors, name, N=512):
    return LinearSegmentedColormap.from_list(name, colors, N=N)

# Qi diverging — deep blue → white → hot red-magenta
CMAP_QI_DIV = _build_cmap([
    "#08306b", "#2171b5", "#6baed6", "#c6dbef",
    "#f7f7f7",
    "#fcbba1", "#fb6a4a", "#cb181d", "#67000d",
], "ast_qi_div")

# Qi sequential — deep teal → electric cyan → warm gold → hot magenta
CMAP_QI_SEQ = _build_cmap([
    "#0a0a2e", "#0b3d6b", "#0891b2", "#06d6a0",
    "#fbbf24", "#f97316", "#ef4444", "#db2777",
], "ast_qi_seq")

# Qsct diverging — purple → white → orange
CMAP_QSCT_DIV = _build_cmap([
    "#3b0764", "#7c3aed", "#a78bfa", "#ddd6fe",
    "#f7f7f7",
    "#fed7aa", "#fb923c", "#ea580c", "#7c2d12",
], "ast_qsct_div")

# Qsct sequential — midnight → purple → neon pink → solar orange
CMAP_QSCT_SEQ = _build_cmap([
    "#0f0720", "#3b0764", "#7c3aed", "#c026d3",
    "#f43f5e", "#fb923c", "#fde047", "#fefce8",
], "ast_qsct_seq")

# Uncertainty — black → indigo → crimson → yellow
CMAP_STD = _build_cmap([
    "#000000", "#1e1b4b", "#4c1d95", "#9333ea",
    "#dc2626", "#f97316", "#facc15", "#fefce8",
], "ast_std")

# Depth slice — deep ocean → teal → lime → hot
CMAP_DEPTH = _build_cmap([
    "#000814", "#001d3d", "#003566", "#0077b6",
    "#00b4d8", "#43aa8b", "#90be6d", "#f9c74f", "#f8961e",
], "ast_depth")

# Map for diverging cross-sections per field
CMAPS_DIV = {"Qi": CMAP_QI_DIV, "Qsct": CMAP_QSCT_DIV}
CMAPS_SEQ = {"Qi": CMAP_QI_SEQ, "Qsct": CMAP_QSCT_SEQ}


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
ROOT = r"C:\Users\shahe\Desktop\Kirkuk SAC files\ast_outputs_kirkuk_v4"
FIG_DIR = os.path.join(ROOT, "cross_sections_v4")
os.makedirs(FIG_DIR, exist_ok=True)

DPI_SAVE = 480
DPI_FIG  = 400

# Frequency bands (for human labels)
FREQ_BANDS = [(0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0)]

def _band_label(b):
    if 0 <= b < len(FREQ_BANDS):
        lo, hi = FREQ_BANDS[b]
        return f"{lo}–{hi} Hz"
    return f"Band {b}"

# Palette
_PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
    "#f4a261", "#264653", "#a855f7", "#06b6d4",
]

plt.rcParams.update({
    "figure.dpi":           DPI_FIG,
    "savefig.dpi":          DPI_SAVE,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "#fafafa",
    "savefig.pad_inches":   0.18,
    "font.family":          "sans-serif",
    "font.sans-serif":      ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":            12,
    "font.weight":          "medium",
    "axes.titlesize":       15,
    "axes.titleweight":     "bold",
    "axes.titlepad":        12,
    "axes.labelsize":       13,
    "axes.labelweight":     "bold",
    "axes.labelpad":        7,
    "axes.linewidth":       1.3,
    "axes.edgecolor":       "#2f2f2f",
    "axes.facecolor":       "#fafafa",
    "axes.prop_cycle":      matplotlib.cycler(color=_PALETTE),
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "xtick.major.width":    1.1,
    "ytick.major.width":    1.1,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "grid.linestyle":       ":",
    "grid.alpha":           0.3,
    "grid.linewidth":       0.7,
    "lines.linewidth":      2.4,
    "legend.fontsize":      10.5,
    "legend.frameon":       True,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#444444",
    "legend.fancybox":      True,
    "legend.shadow":        True,
    "figure.facecolor":     "#fafafa",
})

# Kirkuk reference
KIRKUK_LON, KIRKUK_LAT = 44.39, 35.47

# City markers
MARKERS = {
    "Kirkuk":       (44.39, 35.47),
    "Sulaymaniyah": (45.44, 35.56),
    "Mosul":        (43.15, 36.34),
    "Tikrit":       (43.68, 34.60),
    "Erbil":        (44.01, 36.19),
    "Khanaqin":     (45.39, 34.35),
    "Dukan":        (44.95, 35.95),
}

# Profile definitions
PROFILES = {
    "EW_Kirkuk": {
        "label":     "E–W through Kirkuk",
        "lat":       35.45,
        "direction": "EW",
        "lon_range": (40.5, 47.5),
        "color":     "#e63946",
    },
    "NS_Kirkuk": {
        "label":     "N–S through Kirkuk",
        "lon":       44.44,
        "direction": "NS",
        "lat_range": (31.0, 38.5),
        "color":     "#2a9d8f",
    },
    "Zagros_Diagonal": {
        "label":     "NW–SE Zagros Front",
        "start":     (42.0, 37.5),
        "end":       (47.0, 33.0),
        "direction": "DIAG",
        "color":     "#f4a261",
    },
}

# Subregion depth ranges for annotation
SUBREGION_DEPTHS = {
    "Kirkuk Embayment":      (0, 15),
    "Zagros Foothills":      (0, 15),
    "Hamrin–Makhul":         (0, 15),
    "Kirkuk Deep Crust":     (10, 15),
}


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING (with bounds guard)
# ═══════════════════════════════════════════════════════════════
def load_voxel_csv(filepath):
    """Load a voxel CSV and return structured grid arrays."""
    print(f"  Loading {os.path.basename(filepath)}...")
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except Exception:
            continue
    else:
        print(f"  ERROR: cannot read {filepath}")
        return None

    lons   = np.sort(df["lon"].unique())
    lats   = np.sort(df["lat"].unique())
    depths = np.sort(df["depth_km"].unique())
    bands  = sorted(df["band_index"].unique())

    nx, ny, nz, nb = len(lons), len(lats), len(depths), len(bands)
    print(f"    Grid: {nx}×{ny}×{nz}, {nb} bands")
    print(f"    Lon:   {lons[0]:.2f}–{lons[-1]:.2f}°E")
    print(f"    Lat:   {lats[0]:.2f}–{lats[-1]:.2f}°N")
    print(f"    Depth: {depths[0]:.1f}–{depths[-1]:.1f} km")

    vol = np.full((nb, nz, ny, nx), np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        bi = int(row["band_index"])
        ix = np.searchsorted(lons, row["lon"])
        iy = np.searchsorted(lats, row["lat"])
        iz = np.searchsorted(depths, row["depth_km"])
        # Clamp indices to valid range
        ix = min(ix, nx - 1)
        iy = min(iy, ny - 1)
        iz = min(iz, nz - 1)
        if 0 <= bi < nb:
            vol[bi, iz, iy, ix] = row["value"]

    return {
        "vol": vol, "lons": lons, "lats": lats,
        "depths": depths, "bands": bands,
    }


# ═══════════════════════════════════════════════════════════════
#  CROSS-SECTION EXTRACTION
# ═══════════════════════════════════════════════════════════════
def extract_ew_section(data, band, lat_target):
    """Extract E-W vertical slice at fixed latitude."""
    lats = data["lats"]
    iy = int(np.clip(np.argmin(np.abs(lats - lat_target)), 0, len(lats) - 1))
    actual_lat = lats[iy]
    section = data["vol"][band, :, iy, :]  # [nz, nx]
    return section, data["lons"], data["depths"], actual_lat


def extract_ns_section(data, band, lon_target):
    """Extract N-S vertical slice at fixed longitude."""
    lons = data["lons"]
    ix = int(np.clip(np.argmin(np.abs(lons - lon_target)), 0, len(lons) - 1))
    actual_lon = lons[ix]
    section = data["vol"][band, :, :, ix]  # [nz, ny]
    return section, data["lats"], data["depths"], actual_lon


def extract_diagonal_section(data, band, start, end, npoints=250):
    """
    Extract arbitrary diagonal vertical slice from start to end.
    Returns: section [nz, npoints], distances, depths, line_lons, line_lats.
    """
    lons = data["lons"]
    lats = data["lats"]
    depths = data["depths"]
    vol_b = data["vol"][band]  # [nz, ny, nx]

    t = np.linspace(0, 1, npoints)
    line_lons = start[0] + (end[0] - start[0]) * t
    line_lats = start[1] + (end[1] - start[1]) * t

    # Distances along profile (km)
    dists = np.zeros(npoints)
    for i in range(1, npoints):
        dlat = (line_lats[i] - line_lats[i - 1]) * 111.0
        dlon = ((line_lons[i] - line_lons[i - 1]) * 111.0
                * np.cos(np.radians(0.5 * (line_lats[i] + line_lats[i - 1]))))
        dists[i] = dists[i - 1] + np.sqrt(dlat ** 2 + dlon ** 2)

    section = np.full((len(depths), npoints), np.nan)
    for iz in range(len(depths)):
        try:
            interp = RegularGridInterpolator(
                (lats, lons), vol_b[iz],
                method="linear", bounds_error=False, fill_value=np.nan,
            )
            pts = np.column_stack([line_lats, line_lons])
            section[iz, :] = interp(pts)
        except Exception:
            pass

    return section, dists, depths, line_lons, line_lats


def get_clim(section, percentile=2):
    """Symmetric color limits from percentile, with zero guard."""
    finite = section[np.isfinite(section)]
    if len(finite) == 0:
        return -0.1, 0.1
    vmin = np.percentile(finite, percentile)
    vmax = np.percentile(finite, 100 - percentile)
    vlim = max(abs(vmin), abs(vmax))
    if vlim < 1e-12:
        vlim = max(abs(finite.min()), abs(finite.max()))
    if vlim < 1e-12:
        vlim = 0.1
    return -vlim, vlim


def _safe_twoslope(vmin, vmax, vcenter=0):
    """Create TwoSlopeNorm with safety guards."""
    if vmin >= vcenter:
        vmin = vcenter - max(0.01, abs(vmax - vcenter))
    if vmax <= vcenter:
        vmax = vcenter + max(0.01, abs(vcenter - vmin))
    if vmin >= vmax:
        vmin, vmax = -0.1, 0.1
    return TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)


# ═══════════════════════════════════════════════════════════════
#  HELPER: smart colorbar, contours, markers
# ═══════════════════════════════════════════════════════════════
def _smart_colorbar(im, ax, label):
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    cbar.ax.yaxis.set_major_formatter(fmt)
    return cbar


def _add_section_contours(ax, x, y, data, n_levels=8, color="white"):
    """Add contour overlay on a 2D section."""
    finite = data[np.isfinite(data)]
    if finite.size < 20:
        return
    vmin = np.percentile(finite, 3)
    vmax = np.percentile(finite, 97)
    if abs(vmax - vmin) < 1e-12:
        return
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(vmin, vmax, n_levels)
    try:
        cs = ax.contour(X, Y, data, levels=levels, colors=color,
                        linewidths=0.4, alpha=0.45)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.3g")
    except Exception:
        pass


def _plot_city_markers_ew(ax, actual_lat, lon_range, depths):
    """Plot city markers on E-W sections."""
    for name, (mlon, mlat) in MARKERS.items():
        if (abs(mlat - actual_lat) < 0.5
                and lon_range[0] <= mlon <= lon_range[1]):
            color = "#ffd700" if name == "Kirkuk" else "#333333"
            ms = 9 if name == "Kirkuk" else 6
            marker = "*" if name == "Kirkuk" else "v"
            ax.plot(mlon, 0, marker, color=color, markersize=ms,
                    markeredgecolor="#1a1a1a", markeredgewidth=0.6,
                    zorder=7)
            ax.text(
                mlon, -0.5, name, ha="center", fontsize=7,
                fontweight="bold", color="#b22222" if name == "Kirkuk" else "#264653",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            )
    # Kirkuk vertical line
    if lon_range[0] <= KIRKUK_LON <= lon_range[1]:
        ax.axvline(KIRKUK_LON, color="#ffd700", lw=2, ls="--",
                   alpha=0.7, zorder=3)


def _plot_city_markers_ns(ax, actual_lon, lat_range, depths):
    """Plot city markers on N-S sections."""
    for name, (mlon, mlat) in MARKERS.items():
        if (abs(mlon - actual_lon) < 0.5
                and lat_range[0] <= mlat <= lat_range[1]):
            color = "#ffd700" if name == "Kirkuk" else "#333333"
            ms = 9 if name == "Kirkuk" else 6
            marker = "*" if name == "Kirkuk" else "v"
            ax.plot(mlat, 0, marker, color=color, markersize=ms,
                    markeredgecolor="#1a1a1a", markeredgewidth=0.6,
                    zorder=7)
            ax.text(
                mlat, -0.5, name, ha="center", fontsize=7,
                fontweight="bold", color="#b22222" if name == "Kirkuk" else "#264653",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            )
    if lat_range[0] <= KIRKUK_LAT <= lat_range[1]:
        ax.axvline(KIRKUK_LAT, color="#ffd700", lw=2, ls="--",
                   alpha=0.7, zorder=3)


def _plot_diagonal_markers(ax, dists, line_lons, line_lats, depths):
    """Plot city markers along diagonal profile."""
    for name, (mlon, mlat) in MARKERS.items():
        d2 = np.sqrt((line_lons - mlon) ** 2 + (line_lats - mlat) ** 2)
        if d2.min() < 0.5:
            idx = np.argmin(d2)
            color = "#ffd700" if name == "Kirkuk" else "#aaaaaa"
            ax.axvline(dists[idx], color=color, lw=1.5, ls="--",
                       alpha=0.65, zorder=3)
            ax.text(
                dists[idx], depths[0] - 0.3, name, ha="center",
                fontsize=7, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            )


def _depth_mean_panel(ax, x_arr, section, prof_color, xlabel, ylabel):
    """Gradient-filled depth-mean curve on bottom sub-panel."""
    mean_val = np.nanmean(section, axis=0)
    ax.fill_between(x_arr, 0, mean_val, alpha=0.25, color=prof_color)
    ax.fill_between(
        x_arr, mean_val,
        where=(mean_val >= 0), interpolate=True,
        alpha=0.15, color="#e63946",
    )
    ax.fill_between(
        x_arr, mean_val,
        where=(mean_val < 0), interpolate=True,
        alpha=0.15, color="#457b9d",
    )
    ax.plot(x_arr, mean_val, color=prof_color, lw=2.4)
    ax.axhline(0, color="#888888", lw=0.7, ls="--")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())


# ═══════════════════════════════════════════════════════════════
#  INDIVIDUAL VERTICAL SECTIONS
# ═══════════════════════════════════════════════════════════════
def plot_ew_section(data, data_std, field, band, prof):
    """Enhanced E-W cross-section with contours and gradient fill."""
    section, lons, depths, actual_lat = extract_ew_section(
        data, band, prof["lat"]
    )
    vmin, vmax = get_clim(section)
    freq_label = _band_label(band)
    cmap = CMAPS_DIV.get(field, CMAP_QI_DIV)

    fig, axes = plt.subplots(
        2, 1, figsize=(15, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1], "hspace": 0.06},
    )

    # ── Main section ──
    ax = axes[0]
    norm = _safe_twoslope(vmin, vmax)
    im = ax.pcolormesh(
        lons, depths, section, cmap=cmap, norm=norm,
        shading="gouraud", rasterized=True,
    )
    _add_section_contours(ax, lons, depths, section, n_levels=10)
    ax.set_ylabel("Depth (km)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_title(
        f"{field}$^{{-1}}$ — E–W Section at {actual_lat:.2f}°N  ·  "
        f"{freq_label}",
        fontsize=14, fontweight="bold", pad=14,
    )
    _plot_city_markers_ew(ax, actual_lat, prof["lon_range"], depths)
    _smart_colorbar(im, ax, f"{field}$^{{-1}}$")
    ax.set_xlim(prof["lon_range"])
    ax.grid(True, alpha=0.15)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # ── Depth-mean panel ──
    _depth_mean_panel(
        axes[1], lons, section, prof["color"],
        "Longitude (°E)", f"Mean {field}$^{{-1}}$",
    )
    axes[1].set_xlim(prof["lon_range"])

    out = os.path.join(FIG_DIR, f"{field}_EW_section_band{band}.png")
    fig.savefig(out, dpi=DPI_SAVE, facecolor="#fafafa")
    plt.close(fig)
    print(f"    [OK] {out}")


def plot_ns_section(data, data_std, field, band, prof):
    """Enhanced N-S cross-section."""
    section, lats, depths, actual_lon = extract_ns_section(
        data, band, prof["lon"]
    )
    vmin, vmax = get_clim(section)
    freq_label = _band_label(band)
    cmap = CMAPS_DIV.get(field, CMAP_QI_DIV)

    fig, axes = plt.subplots(
        2, 1, figsize=(15, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1], "hspace": 0.06},
    )

    ax = axes[0]
    norm = _safe_twoslope(vmin, vmax)
    im = ax.pcolormesh(
        lats, depths, section, cmap=cmap, norm=norm,
        shading="gouraud", rasterized=True,
    )
    _add_section_contours(ax, lats, depths, section, n_levels=10)
    ax.set_ylabel("Depth (km)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_title(
        f"{field}$^{{-1}}$ — N–S Section at {actual_lon:.2f}°E  ·  "
        f"{freq_label}",
        fontsize=14, fontweight="bold", pad=14,
    )
    _plot_city_markers_ns(ax, actual_lon, prof["lat_range"], depths)
    _smart_colorbar(im, ax, f"{field}$^{{-1}}$")
    ax.set_xlim(prof["lat_range"])
    ax.grid(True, alpha=0.15)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    _depth_mean_panel(
        axes[1], lats, section, prof["color"],
        "Latitude (°N)", f"Mean {field}$^{{-1}}$",
    )
    axes[1].set_xlim(prof["lat_range"])

    out = os.path.join(FIG_DIR, f"{field}_NS_section_band{band}.png")
    fig.savefig(out, dpi=DPI_SAVE, facecolor="#fafafa")
    plt.close(fig)
    print(f"    [OK] {out}")


def plot_diagonal_section(data, data_std, field, band, prof):
    """Enhanced diagonal cross-section along Zagros front."""
    section, dists, depths, line_lons, line_lats = extract_diagonal_section(
        data, band, prof["start"], prof["end"],
    )
    vmin, vmax = get_clim(section)
    freq_label = _band_label(band)
    cmap = CMAPS_DIV.get(field, CMAP_QI_DIV)

    fig, axes = plt.subplots(
        2, 1, figsize=(15, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1], "hspace": 0.06},
    )

    ax = axes[0]
    norm = _safe_twoslope(vmin, vmax)
    im = ax.pcolormesh(
        dists, depths, section, cmap=cmap, norm=norm,
        shading="gouraud", rasterized=True,
    )
    _add_section_contours(ax, dists, depths, section, n_levels=10)
    ax.set_ylabel("Depth (km)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_title(
        f"{field}$^{{-1}}$ — Diagonal ({prof['start'][0]:.0f}°E,"
        f"{prof['start'][1]:.0f}°N) → ({prof['end'][0]:.0f}°E,"
        f"{prof['end'][1]:.0f}°N)  ·  {freq_label}",
        fontsize=13, fontweight="bold", pad=14,
    )
    _plot_diagonal_markers(ax, dists, line_lons, line_lats, depths)
    _smart_colorbar(im, ax, f"{field}$^{{-1}}$")
    ax.grid(True, alpha=0.15)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Start/end labels
    ax.text(
        dists[2], depths[-1] + 0.4,
        f"NW ({prof['start'][0]:.0f}°E, {prof['start'][1]:.0f}°N)",
        fontsize=8.5, fontweight="bold", color="#264653",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )
    ax.text(
        dists[-3], depths[-1] + 0.4,
        f"SE ({prof['end'][0]:.0f}°E, {prof['end'][1]:.0f}°N)",
        fontsize=8.5, fontweight="bold", color="#264653", ha="right",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

    _depth_mean_panel(
        axes[1], dists, section, prof["color"],
        "Distance along profile (km)", f"Mean {field}$^{{-1}}$",
    )

    out = os.path.join(FIG_DIR, f"{field}_Zagros_diagonal_band{band}.png")
    fig.savefig(out, dpi=DPI_SAVE, facecolor="#fafafa")
    plt.close(fig)
    print(f"    [OK] {out}")


# ═══════════════════════════════════════════════════════════════
#  COMBINED PANEL: DEPTH SLICE + 3 CROSS-SECTIONS
# ═══════════════════════════════════════════════════════════════
def plot_combined_panel(data, field, band, depth_km=5.0):
    """4-panel: depth slice with profile lines + 3 sections."""
    lons = data["lons"]
    lats = data["lats"]
    depths = data["depths"]
    iz = int(np.clip(np.argmin(np.abs(depths - depth_km)), 0, len(depths) - 1))
    actual_depth = depths[iz]
    hslice = data["vol"][band, iz]

    ew = PROFILES["EW_Kirkuk"]
    ns = PROFILES["NS_Kirkuk"]
    dg = PROFILES["Zagros_Diagonal"]
    freq_label = _band_label(band)
    cmap_div = CMAPS_DIV.get(field, CMAP_QI_DIV)
    cmap_seq = CMAPS_SEQ.get(field, CMAP_QI_SEQ)

    sec_ew, lons_ew, dep_ew, lat_ew = extract_ew_section(data, band, ew["lat"])
    sec_ns, lats_ns, dep_ns, lon_ns = extract_ns_section(data, band, ns["lon"])
    sec_dg, dists_dg, dep_dg, dg_lons, dg_lats = extract_diagonal_section(
        data, band, dg["start"], dg["end"],
    )

    fig = plt.figure(figsize=(22, 17))
    gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.24)

    # ── Top-left: depth slice + profile traces ──
    ax0 = fig.add_subplot(gs[0, 0])
    finite_h = hslice[np.isfinite(hslice)]
    if finite_h.size > 0:
        vmin_h = np.percentile(finite_h, 1)
        vmax_h = np.percentile(finite_h, 99)
    else:
        vmin_h, vmax_h = -0.1, 0.1
    im0 = ax0.pcolormesh(
        lons, lats, hslice, cmap=cmap_seq,
        vmin=vmin_h, vmax=vmax_h,
        shading="gouraud", rasterized=True,
    )
    # Profile traces
    ax0.axhline(lat_ew, color=ew["color"], lw=2.5, label="E–W", zorder=4)
    ax0.axvline(lon_ns, color=ns["color"], lw=2.5, label="N–S", zorder=4)
    ax0.plot(
        [dg["start"][0], dg["end"][0]],
        [dg["start"][1], dg["end"][1]],
        color=dg["color"], lw=2.5, label="Zagros diag.", zorder=4,
    )
    # Kirkuk
    ax0.plot(
        KIRKUK_LON, KIRKUK_LAT, "*", color="#ffd700", markersize=14,
        markeredgecolor="#1a1a1a", markeredgewidth=0.8, zorder=6,
    )
    ax0.text(
        KIRKUK_LON + 0.2, KIRKUK_LAT + 0.15, "Kirkuk",
        fontsize=9, fontweight="bold", color="#b22222",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )
    ax0.set_xlabel("Longitude (°E)")
    ax0.set_ylabel("Latitude (°N)")
    ax0.set_title(
        f"Depth slice {actual_depth:.0f} km — profile locations",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax0.set_xlim(40.5, 47.5)
    ax0.set_ylim(31, 38.5)
    ax0.legend(loc="lower left", fontsize=9.5, framealpha=0.92)
    _smart_colorbar(im0, ax0, f"{field}$^{{-1}}$")
    ax0.grid(True, alpha=0.15)
    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax0.yaxis.set_minor_locator(AutoMinorLocator())

    # ── Top-right: E-W ──
    ax1 = fig.add_subplot(gs[0, 1])
    v1, v2 = get_clim(sec_ew)
    im1 = ax1.pcolormesh(
        lons_ew, dep_ew, sec_ew, cmap=cmap_div,
        norm=_safe_twoslope(v1, v2), shading="gouraud", rasterized=True,
    )
    _add_section_contours(ax1, lons_ew, dep_ew, sec_ew, n_levels=8)
    ax1.invert_yaxis()
    ax1.axvline(KIRKUK_LON, color="#ffd700", lw=1.5, ls="--", alpha=0.7)
    ax1.axhline(actual_depth, color="white", lw=1.2, ls=":", alpha=0.6)
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Depth (km)")
    ax1.set_title(
        f"E–W at {lat_ew:.2f}°N", color=ew["color"],
        fontsize=13, fontweight="bold", pad=10,
    )
    ax1.set_xlim(ew["lon_range"])
    _smart_colorbar(im1, ax1, f"{field}$^{{-1}}$")
    ax1.grid(True, alpha=0.15)

    # ── Bottom-left: N-S ──
    ax2 = fig.add_subplot(gs[1, 0])
    v1, v2 = get_clim(sec_ns)
    im2 = ax2.pcolormesh(
        lats_ns, dep_ns, sec_ns, cmap=cmap_div,
        norm=_safe_twoslope(v1, v2), shading="gouraud", rasterized=True,
    )
    _add_section_contours(ax2, lats_ns, dep_ns, sec_ns, n_levels=8)
    ax2.invert_yaxis()
    ax2.axvline(KIRKUK_LAT, color="#ffd700", lw=1.5, ls="--", alpha=0.7)
    ax2.axhline(actual_depth, color="white", lw=1.2, ls=":", alpha=0.6)
    ax2.set_xlabel("Latitude (°N)")
    ax2.set_ylabel("Depth (km)")
    ax2.set_title(
        f"N–S at {lon_ns:.2f}°E", color=ns["color"],
        fontsize=13, fontweight="bold", pad=10,
    )
    ax2.set_xlim(ns["lat_range"])
    _smart_colorbar(im2, ax2, f"{field}$^{{-1}}$")
    ax2.grid(True, alpha=0.15)

    # ── Bottom-right: diagonal ──
    ax3 = fig.add_subplot(gs[1, 1])
    v1, v2 = get_clim(sec_dg)
    im3 = ax3.pcolormesh(
        dists_dg, dep_dg, sec_dg, cmap=cmap_div,
        norm=_safe_twoslope(v1, v2), shading="gouraud", rasterized=True,
    )
    _add_section_contours(ax3, dists_dg, dep_dg, sec_dg, n_levels=8)
    ax3.invert_yaxis()
    ax3.axhline(actual_depth, color="white", lw=1.2, ls=":", alpha=0.6)
    ax3.set_xlabel("Distance (km)")
    ax3.set_ylabel("Depth (km)")
    ax3.set_title(
        "NW–SE Zagros Diagonal", color=dg["color"],
        fontsize=13, fontweight="bold", pad=10,
    )
    ax3.text(
        0.02, 0.03,
        f"NW ({dg['start'][0]:.0f}°E, {dg['start'][1]:.0f}°N)",
        transform=ax3.transAxes, fontsize=8.5, fontweight="bold",
        va="bottom", color="#264653",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )
    ax3.text(
        0.98, 0.03,
        f"SE ({dg['end'][0]:.0f}°E, {dg['end'][1]:.0f}°N)",
        transform=ax3.transAxes, fontsize=8.5, fontweight="bold",
        va="bottom", ha="right", color="#264653",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )
    _smart_colorbar(im3, ax3, f"{field}$^{{-1}}$")
    ax3.grid(True, alpha=0.15)

    fig.suptitle(
        f"{field} Attenuation Tomography — Kirkuk Region  ·  "
        f"{freq_label}  ·  Ref. depth {actual_depth:.0f} km",
        fontsize=17, fontweight="bold", y=0.99,
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="#ffffffee",
            edgecolor="#2f2f2f", linewidth=1.6,
        ),
    )

    out = os.path.join(
        FIG_DIR,
        f"{field}_combined_panel_band{band}_z{actual_depth:.0f}km.png",
    )
    fig.savefig(out, dpi=DPI_SAVE, facecolor="#fafafa", pad_inches=0.3)
    plt.close(fig)
    print(f"    [OK] {out}")


# ═══════════════════════════════════════════════════════════════
#  3D FENCE DIAGRAM (enhanced dual-view)
# ═══════════════════════════════════════════════════════════════
def plot_3d_fence(data, field, band, depth_km=5.0):
    """3D perspective with depth slice + E-W/N-S vertical planes."""
    lons = data["lons"]
    lats = data["lats"]
    depths = data["depths"]
    iz = int(np.clip(np.argmin(np.abs(depths - depth_km)), 0, len(depths) - 1))
    actual_depth = depths[iz]
    freq_label = _band_label(band)

    ew = PROFILES["EW_Kirkuk"]
    ns = PROFILES["NS_Kirkuk"]

    sec_ew, lons_ew, dep_ew, lat_ew = extract_ew_section(data, band, ew["lat"])
    sec_ns, lats_ns, dep_ns, lon_ns = extract_ns_section(data, band, ns["lon"])
    hslice = data["vol"][band, iz]

    # Collect all finite values for color limits
    all_finite = []
    for arr in [sec_ew, sec_ns, hslice]:
        f = arr[np.isfinite(arr)]
        if f.size > 0:
            all_finite.append(f)
    if not all_finite:
        print(f"    [SKIP] 3D fence: no finite data")
        return
    finite_all = np.concatenate(all_finite)
    vlim = np.percentile(np.abs(finite_all), 95)
    if vlim < 1e-12:
        vlim = 0.1
    norm = _safe_twoslope(-vlim, vlim)
    cmap = CMAPS_DIV.get(field, CMAP_QI_DIV)

    # Subsample for 3D performance
    step_h_lons = max(1, len(lons) // 60)
    step_h_lats = max(1, len(lats) // 60)
    step_ew = max(1, len(lons_ew) // 60)
    step_ns = max(1, len(lats_ns) // 60)
    step_v = max(1, len(depths) // 12)

    for view_name, elev, azim in [("view1", 28, -50), ("view2", 18, -130)]:
        fig = plt.figure(figsize=(17, 13))
        ax = fig.add_subplot(111, projection='3d')

        # ── Horizontal depth slice ──
        lon_sub = lons[::step_h_lons]
        lat_sub = lats[::step_h_lats]
        hs_sub = hslice[::step_h_lats, :][:, ::step_h_lons]
        LON, LAT = np.meshgrid(lon_sub, lat_sub)
        Z_slice = np.full_like(LON, actual_depth)
        colors_h = cmap(norm(np.nan_to_num(hs_sub, nan=0)))
        ax.plot_surface(
            LON, LAT, Z_slice, facecolors=colors_h,
            alpha=0.65, shade=False, zorder=1,
        )

        # ── E-W vertical plane ──
        ew_lons_sub = lons_ew[::step_ew]
        ew_dep_sub = dep_ew[::step_v]
        ew_sec_sub = sec_ew[::step_v, :][:, ::step_ew]
        EW_LON, EW_DEP = np.meshgrid(ew_lons_sub, ew_dep_sub)
        EW_LAT = np.full_like(EW_LON, lat_ew)
        colors_ew = cmap(norm(np.nan_to_num(ew_sec_sub, nan=0)))
        ax.plot_surface(
            EW_LON, EW_LAT, EW_DEP, facecolors=colors_ew,
            alpha=0.82, shade=False, zorder=2,
        )

        # ── N-S vertical plane ──
        ns_lats_sub = lats_ns[::step_ns]
        ns_dep_sub = dep_ns[::step_v]
        ns_sec_sub = sec_ns[::step_v, :][:, ::step_ns]
        NS_LAT, NS_DEP = np.meshgrid(ns_lats_sub, ns_dep_sub)
        NS_LON = np.full_like(NS_LAT, lon_ns)
        colors_ns = cmap(norm(np.nan_to_num(ns_sec_sub, nan=0)))
        ax.plot_surface(
            NS_LON, NS_LAT, NS_DEP, facecolors=colors_ns,
            alpha=0.82, shade=False, zorder=3,
        )

        # Kirkuk marker
        ax.scatter(
            [KIRKUK_LON], [KIRKUK_LAT], [0], marker="*",
            s=250, c="#ffd700", edgecolors="#1a1a1a", linewidths=0.8,
            zorder=10,
        )
        ax.text(
            KIRKUK_LON, KIRKUK_LAT, -0.8, "Kirkuk",
            fontsize=9, fontweight="bold", color="#b22222", zorder=10,
        )

        # Profile traces on surface
        ax.plot(
            [ew["lon_range"][0], ew["lon_range"][1]], [lat_ew, lat_ew], [0, 0],
            color=ew["color"], lw=2.5, zorder=5,
        )
        ax.plot(
            [lon_ns, lon_ns], [ns["lat_range"][0], ns["lat_range"][1]], [0, 0],
            color=ns["color"], lw=2.5, zorder=5,
        )

        ax.set_xlabel("Longitude (°E)", labelpad=12, fontsize=12)
        ax.set_ylabel("Latitude (°N)", labelpad=12, fontsize=12)
        ax.set_zlabel("Depth (km)", labelpad=10, fontsize=12)
        ax.invert_zaxis()
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(40.5, 47.5)
        ax.set_ylim(31, 38.5)
        ax.set_zlim(depths[-1], 0)

        ax.set_title(
            f"3D Fence — {field}$^{{-1}}$  ·  {freq_label}\n"
            f"Depth slice {actual_depth:.0f} km + E–W/N–S through Kirkuk",
            fontsize=14, fontweight="bold", pad=20,
        )

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.08, aspect=20)
        cbar.set_label(f"{field}$^{{-1}}$", fontweight="bold", fontsize=12)

        out = os.path.join(
            FIG_DIR,
            f"{field}_3D_fence_band{band}_z{actual_depth:.0f}km_{view_name}.png",
        )
        fig.savefig(out, dpi=DPI_SAVE, facecolor="#fafafa", pad_inches=0.3)
        plt.close(fig)
        print(f"    [OK] {out}")


# ═══════════════════════════════════════════════════════════════
#  MULTI-DEPTH E-W COMPARISON
# ═══════════════════════════════════════════════════════════════
def plot_depth_comparison_ew(data, field, band, prof):
    """E-W depth-mean curves at selected depths, stacked."""
    _, lons_ew, dep_ew, lat_ew = extract_ew_section(data, band, prof["lat"])
    freq_label = _band_label(band)

    # Pick depths that actually exist
    target_depths = [0, 3, 7, 10, 14]
    actual_izs = []
    for td in target_depths:
        iz = int(np.clip(np.argmin(np.abs(dep_ew - td)), 0, len(dep_ew) - 1))
        if iz not in [a[0] for a in actual_izs]:
            actual_izs.append((iz, dep_ew[iz]))
    if not actual_izs:
        return

    sec_all, _, _, _ = extract_ew_section(data, band, prof["lat"])
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, len(actual_izs)))

    fig, axes = plt.subplots(
        len(actual_izs), 1,
        figsize=(15, 3.0 * len(actual_izs)),
        sharex=True,
    )
    if len(actual_izs) == 1:
        axes = [axes]

    for i, (iz, dep_val) in enumerate(actual_izs):
        ax = axes[i]
        vals = sec_all[iz, :]
        ax.fill_between(
            lons_ew, 0, vals, alpha=0.2, color=colors[i],
        )
        ax.fill_between(
            lons_ew, vals, where=(vals >= 0),
            interpolate=True, alpha=0.15, color="#e63946",
        )
        ax.fill_between(
            lons_ew, vals, where=(vals < 0),
            interpolate=True, alpha=0.15, color="#457b9d",
        )
        ax.plot(lons_ew, vals, color=colors[i], lw=2.4)
        ax.axhline(0, color="#888888", lw=0.7, ls="--")
        if prof["lon_range"][0] <= KIRKUK_LON <= prof["lon_range"][1]:
            ax.axvline(KIRKUK_LON, color="#ffd700", lw=1.5, ls="--",
                       alpha=0.6)
        ax.set_ylabel(
            f"{dep_val:.0f} km", fontsize=12, fontweight="bold",
        )
        ax.set_xlim(prof["lon_range"])
        ax.grid(True, alpha=0.15)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if i == 0:
            ax.set_title(
                f"{field}$^{{-1}}$ E–W at {lat_ew:.2f}°N — "
                f"depth comparison  ·  {freq_label}",
                fontsize=14, fontweight="bold", pad=10,
            )

    axes[-1].set_xlabel("Longitude (°E)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = os.path.join(
        FIG_DIR, f"{field}_EW_depth_comparison_band{band}.png"
    )
    fig.savefig(out, dpi=DPI_SAVE, facecolor="#fafafa")
    plt.close(fig)
    print(f"    [OK] {out}")


# ═══════════════════════════════════════════════════════════════
#  ALL-BANDS E-W COMPARISON
# ═══════════════════════════════════════════════════════════════
def plot_all_bands_ew(data, field, prof):
    """E-W section for all bands stacked vertically."""
    bands = data["bands"]
    cmap = CMAPS_DIV.get(field, CMAP_QI_DIV)

    fig, axes = plt.subplots(
        len(bands), 1,
        figsize=(15, 3.5 * len(bands)),
        sharex=True,
    )
    if len(bands) == 1:
        axes = [axes]

    lat_ew = None
    for i, b in enumerate(bands):
        sec, lons_ew, dep_ew, lat_ew_i = extract_ew_section(
            data, b, prof["lat"]
        )
        lat_ew = lat_ew_i
        vmin, vmax = get_clim(sec)
        freq_label = _band_label(b)

        ax = axes[i]
        norm = _safe_twoslope(vmin, vmax)
        im = ax.pcolormesh(
            lons_ew, dep_ew, sec, cmap=cmap, norm=norm,
            shading="gouraud", rasterized=True,
        )
        _add_section_contours(ax, lons_ew, dep_ew, sec, n_levels=6,
                              color="white")
        ax.invert_yaxis()
        if prof["lon_range"][0] <= KIRKUK_LON <= prof["lon_range"][1]:
            ax.axvline(KIRKUK_LON, color="#ffd700", lw=1.5, ls="--",
                       alpha=0.7)
        ax.set_ylabel("Depth (km)", fontsize=11)
        ax.set_title(
            freq_label, fontsize=12, fontweight="bold", pad=6,
        )
        ax.set_xlim(prof["lon_range"])
        ax.grid(True, alpha=0.12)
        _smart_colorbar(im, ax, f"{field}$^{{-1}}$")

    axes[-1].set_xlabel("Longitude (°E)", fontsize=13, fontweight="bold")
    fig.suptitle(
        f"{field}$^{{-1}}$ E–W Section at "
        f"{lat_ew:.2f}°N — All Frequency Bands",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out = os.path.join(FIG_DIR, f"{field}_EW_all_bands.png")
    fig.savefig(out, dpi=DPI_SAVE, facecolor="#fafafa")
    plt.close(fig)
    print(f"    [OK] {out}")


# ═══════════════════════════════════════════════════════════════
#  UNCERTAINTY CROSS-SECTIONS (NEW)
# ═══════════════════════════════════════════════════════════════
def plot_uncertainty_sections(data, data_std, field, band, prof_ew, prof_ns):
    """Side-by-side mean + std for E-W and N-S."""
    if data_std is None:
        return
    freq_label = _band_label(band)
    cmap_mean = CMAPS_DIV.get(field, CMAP_QI_DIV)

    sec_ew, lons_ew, dep_ew, lat_ew = extract_ew_section(data, band, prof_ew["lat"])
    std_ew, _, _, _ = extract_ew_section(data_std, band, prof_ew["lat"])

    sec_ns, lats_ns, dep_ns, lon_ns = extract_ns_section(data, band, prof_ns["lon"])
    std_ns, _, _, _ = extract_ns_section(data_std, band, prof_ns["lon"])

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    # E-W mean
    ax = axes[0, 0]
    v1, v2 = get_clim(sec_ew)
    im = ax.pcolormesh(
        lons_ew, dep_ew, sec_ew, cmap=cmap_mean,
        norm=_safe_twoslope(v1, v2), shading="gouraud", rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_title(f"E–W Mean at {lat_ew:.2f}°N", fontweight="bold")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(prof_ew["lon_range"])
    _smart_colorbar(im, ax, f"{field}$^{{-1}}$ mean")
    ax.grid(True, alpha=0.15)

    # E-W std
    ax = axes[0, 1]
    im = ax.pcolormesh(
        lons_ew, dep_ew, std_ew, cmap=CMAP_STD,
        shading="gouraud", rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_title(f"E–W Uncertainty at {lat_ew:.2f}°N", fontweight="bold")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(prof_ew["lon_range"])
    _smart_colorbar(im, ax, f"{field}$^{{-1}}$ std")
    ax.grid(True, alpha=0.15)

    # N-S mean
    ax = axes[1, 0]
    v1, v2 = get_clim(sec_ns)
    im = ax.pcolormesh(
        lats_ns, dep_ns, sec_ns, cmap=cmap_mean,
        norm=_safe_twoslope(v1, v2), shading="gouraud", rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_title(f"N–S Mean at {lon_ns:.2f}°E", fontweight="bold")
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(prof_ns["lat_range"])
    _smart_colorbar(im, ax, f"{field}$^{{-1}}$ mean")
    ax.grid(True, alpha=0.15)

    # N-S std
    ax = axes[1, 1]
    im = ax.pcolormesh(
        lats_ns, dep_ns, std_ns, cmap=CMAP_STD,
        shading="gouraud", rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_title(f"N–S Uncertainty at {lon_ns:.2f}°E", fontweight="bold")
    ax.set_xlabel("Latitude (°N)")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(prof_ns["lat_range"])
    _smart_colorbar(im, ax, f"{field}$^{{-1}}$ std")
    ax.grid(True, alpha=0.15)

    fig.suptitle(
        f"{field} Mean & Uncertainty Cross-Sections  ·  {freq_label}",
        fontsize=16, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(
        FIG_DIR, f"{field}_uncertainty_sections_band{band}.png"
    )
    fig.savefig(out, dpi=DPI_SAVE, facecolor="#fafafa")
    plt.close(fig)
    print(f"    [OK] {out}")


# ═══════════════════════════════════════════════════════════════
#  ALL-BANDS N-S COMPARISON (NEW)
# ═══════════════════════════════════════════════════════════════
def plot_all_bands_ns(data, field, prof):
    """N-S section for all bands stacked vertically."""
    bands = data["bands"]
    cmap = CMAPS_DIV.get(field, CMAP_QI_DIV)

    fig, axes = plt.subplots(
        len(bands), 1,
        figsize=(15, 3.5 * len(bands)),
        sharex=True,
    )
    if len(bands) == 1:
        axes = [axes]

    lon_ns = None
    for i, b in enumerate(bands):
        sec, lats_ns, dep_ns, lon_ns_i = extract_ns_section(
            data, b, prof["lon"]
        )
        lon_ns = lon_ns_i
        vmin, vmax = get_clim(sec)
        freq_label = _band_label(b)

        ax = axes[i]
        norm = _safe_twoslope(vmin, vmax)
        im = ax.pcolormesh(
            lats_ns, dep_ns, sec, cmap=cmap, norm=norm,
            shading="gouraud", rasterized=True,
        )
        _add_section_contours(ax, lats_ns, dep_ns, sec, n_levels=6,
                              color="white")
        ax.invert_yaxis()
        if prof["lat_range"][0] <= KIRKUK_LAT <= prof["lat_range"][1]:
            ax.axvline(KIRKUK_LAT, color="#ffd700", lw=1.5, ls="--",
                       alpha=0.7)
        ax.set_ylabel("Depth (km)", fontsize=11)
        ax.set_title(freq_label, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlim(prof["lat_range"])
        ax.grid(True, alpha=0.12)
        _smart_colorbar(im, ax, f"{field}$^{{-1}}$")

    axes[-1].set_xlabel("Latitude (°N)", fontsize=13, fontweight="bold")
    fig.suptitle(
        f"{field}$^{{-1}}$ N–S Section at "
        f"{lon_ns:.2f}°E — All Frequency Bands",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out = os.path.join(FIG_DIR, f"{field}_NS_all_bands.png")
    fig.savefig(out, dpi=DPI_SAVE, facecolor="#fafafa")
    plt.close(fig)
    print(f"    [OK] {out}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  3D Cross-Section Visualizer v4.0 — Kirkuk Tomography")
    print("=" * 60)
    print(f"  Output:  {FIG_DIR}")
    print(f"  DPI:     {DPI_SAVE}")
    print()

    # Load data
    qi_path     = os.path.join(ROOT, "Qi_voxels.csv")
    qs_path     = os.path.join(ROOT, "Qsct_voxels.csv")
    qi_std_path = os.path.join(ROOT, "Qi_std_voxels.csv")
    qs_std_path = os.path.join(ROOT, "Qsct_std_voxels.csv")

    datasets = {}
    for name, path in [("Qi", qi_path), ("Qsct", qs_path)]:
        if os.path.exists(path):
            d = load_voxel_csv(path)
            if d is not None:
                datasets[name] = d
        else:
            print(f"  [SKIP] {path} not found")

    datasets_std = {}
    for name, path in [("Qi", qi_std_path), ("Qsct", qs_std_path)]:
        if os.path.exists(path):
            d = load_voxel_csv(path)
            if d is not None:
                datasets_std[name] = d

    if not datasets:
        print("No data found. Exiting.")
        return

    for field, data in datasets.items():
        data_std = datasets_std.get(field)
        bands = data["bands"]
        print(f"\n{'=' * 50}")
        print(f"  Generating {field} cross-sections")
        print(f"{'=' * 50}")

        # Primary band: 3–6 Hz (band 2) if available
        primary_band = 2 if 2 in bands else bands[0]

        # 1. Individual sections (primary band)
        print(f"\n  [1] Individual sections (band {primary_band})...")
        plot_ew_section(data, data_std, field, primary_band,
                        PROFILES["EW_Kirkuk"])
        plot_ns_section(data, data_std, field, primary_band,
                        PROFILES["NS_Kirkuk"])
        plot_diagonal_section(data, data_std, field, primary_band,
                              PROFILES["Zagros_Diagonal"])

        # 2. Combined 4-panel at multiple depths
        print(f"\n  [2] Combined panels...")
        for dz in [0, 3, 5, 8, 12]:
            plot_combined_panel(data, field, primary_band, depth_km=dz)

        # 3. 3D fence diagram (dual view)
        print(f"\n  [3] 3D fence diagrams...")
        for dz in [3, 7]:
            plot_3d_fence(data, field, primary_band, depth_km=dz)

        # 4. Multi-depth E-W comparison
        print(f"\n  [4] Depth comparison E-W...")
        plot_depth_comparison_ew(data, field, primary_band,
                                 PROFILES["EW_Kirkuk"])

        # 5. All-bands comparisons
        print(f"\n  [5] All-bands comparisons...")
        plot_all_bands_ew(data, field, PROFILES["EW_Kirkuk"])
        plot_all_bands_ns(data, field, PROFILES["NS_Kirkuk"])

        # 6. Uncertainty sections
        print(f"\n  [6] Uncertainty cross-sections...")
        plot_uncertainty_sections(
            data, data_std, field, primary_band,
            PROFILES["EW_Kirkuk"], PROFILES["NS_Kirkuk"],
        )

        # 7. All bands individual sections
        print(f"\n  [7] All bands individual sections...")
        for b in bands:
            if b == primary_band:
                continue
            plot_ew_section(data, data_std, field, b,
                            PROFILES["EW_Kirkuk"])
            plot_ns_section(data, data_std, field, b,
                            PROFILES["NS_Kirkuk"])
            plot_diagonal_section(data, data_std, field, b,
                                  PROFILES["Zagros_Diagonal"])

    print(f"\n{'=' * 60}")
    print(f"  All outputs saved to: {FIG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()


# In[4]:


# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  PUBLICATION FIGURE SUITE v4.0                                     ║
║  Iraq Seismic Attenuation Tomography                               ║
║  Fault Segmentation & Crustal Structure Analysis                   ║
╚══════════════════════════════════════════════════════════════════════╝

Generates all figures for a journal paper on attenuation tomography
of Iraq, focused on fault detection, segmentation, depth extent,
and activity characterization.

Reads:   Kirkuk_Attenuation_Tomography.nc (CF-1.8 NetCDF)
Outputs: ~15 publication-ready PNG figures + summary tables (CSV)

Target journals: JGR Solid Earth, Tectonophysics, BSSA, GJI

Changelog from v3.1:
  ── BUG FIXES ──
   1. TwoSlopeNorm: guarded vmin >= vcenter and vmax <= vcenter
   2. get_sym_clim: guarded vlim == 0 (all-identical values)
   3. extract_section: bounds-checked ix, iy indices
   4. fig_convergence: guarded against missing loss_total variable
   5. fig_depth_profiles: guarded against out-of-bounds ix, iy
   6. table_fault_summary: guarded np.isfinite on ratio/SNR
   7. Colormaps registered with matplotlib to avoid plt.get_cmap crash
   8. Colorbar: ScalarFormatter with proper power limits

  ── VISUALIZATION UPGRADES ──
   • DPI raised to 480 for publication quality
   • 7 custom high-contrast colormaps (registered globally)
   • Contour overlays on all map panels
   • Enhanced typography (serif, heavier weights, better spacing)
   • Bilinear interpolation via 'gouraud' shading
   • City/fault markers with improved halo path effects
   • Subregion depth annotations on cross-sections
   • Minor ticks on all axes (0.2° geographic grid)
   • Consistent colorbar formatting with ScalarFormatter
   • Gradient-filled depth-mean curves on cross-sections
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from scipy.interpolate import RegularGridInterpolator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import (
    LinearSegmentedColormap, TwoSlopeNorm, Normalize,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    MultipleLocator, AutoMinorLocator, ScalarFormatter,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  CUSTOM COLORMAPS — energetic, high-contrast, publication-ready
#  All registered globally so plt.get_cmap() also works.
# ═══════════════════════════════════════════════════════════════

def _make_and_register(colors, name, N=512):
    """Build a LinearSegmentedColormap and register it globally."""
    cmap = LinearSegmentedColormap.from_list(name, colors, N=N)
    try:
        matplotlib.colormaps.register(cmap, name=name, force=True)
    except AttributeError:
        # matplotlib < 3.7 fallback
        try:
            plt.register_cmap(name=name, cmap=cmap)
        except Exception:
            pass
    return cmap

# Qi diverging — ink navy → electric blue → thin white → scarlet → blood
CMAP_QI = _make_and_register([
    "#03071e", "#032174", "#0353a4", "#0466c8", "#48cae4",
    "#f8f9fa",
    "#ffba08", "#f48c06", "#e63946", "#9d0208", "#370617",
], "pub_qi")

# Qsct diverging — abyss purple → neon violet → thin white → fire → mahogany
CMAP_QSCT = _make_and_register([
    "#10002b", "#240046", "#5a189a", "#9d4edd", "#c77dff",
    "#f8f9fa",
    "#ffdd00", "#ff9500", "#ff4800", "#c1121f", "#4a0404",
], "pub_qsct")

# Qt total — deep teal → cyan flash → thin white → hot pink → black cherry
CMAP_QT = _make_and_register([
    "#001524", "#003554", "#006494", "#0582ca", "#00b4d8",
    "#f8f9fa",
    "#fca311", "#f72585", "#b5179e", "#7209b7", "#240046",
], "pub_qt")

# Ratio Qsct/Qi — vivid emerald (Qi-dom) → thin white → blazing amber (Qsct-dom)
CMAP_RATIO = _make_and_register([
    "#004b23", "#007200", "#38b000", "#70e000", "#ccff33",
    "#f8f9fa",
    "#ffe169", "#ffc300", "#ff8800", "#e36414", "#6a040f",
], "pub_ratio")

# Uncertainty — true black → electric indigo → neon magenta → solar yellow → white
CMAP_STD = _make_and_register([
    "#000000", "#14002d", "#4a0078", "#9d00ff",
    "#ff006e", "#ff4d00", "#ffbe0b", "#fffacd",
], "pub_std")

# SNR reliability — black → blood red → amber → neon green → white
CMAP_SNR = _make_and_register([
    "#1a0000", "#800000", "#cc0000", "#ff4500",
    "#ffbf00", "#7fff00", "#00e676", "#00ff88",
], "pub_snr")

# Convergence — midnight → electric blue → cyan → white
CMAP_CONV = _make_and_register([
    "#020024", "#03045e", "#0077b6", "#00b4d8",
    "#48cae4", "#90e0ef", "#caf0f8",
], "pub_conv")


# ═══════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════
NC_PATH = (
    r"C:\Users\shahe\Desktop\Kirkuk SAC files"
    r"\ast_outputs_kirkuk\Kirkuk_Attenuation_Tomography.nc"
)
FIG_DIR = os.path.join(os.path.dirname(NC_PATH), "paper_figures_v4")
TBL_DIR = os.path.join(FIG_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

DPI_SAVE = 480
DPI_FIG  = 400

STROKE      = [pe.withStroke(linewidth=3.0, foreground="white")]
STROKE_THIN = [pe.withStroke(linewidth=2.2, foreground="white")]

# Palette for line plots
_PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
    "#f4a261", "#264653", "#a855f7", "#06b6d4",
    "#d946ef", "#14b8a6",
]

plt.rcParams.update({
    # Resolution
    "figure.dpi":           DPI_FIG,
    "savefig.dpi":          DPI_SAVE,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",
    "savefig.pad_inches":   0.15,

    # Typography — serif for journal submission
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size":            11.5,
    "font.weight":          "medium",

    # Axes
    "axes.titlesize":       14,
    "axes.titleweight":     "bold",
    "axes.titlepad":        12,
    "axes.labelsize":       12.5,
    "axes.labelweight":     "bold",
    "axes.labelpad":        7,
    "axes.linewidth":       1.3,
    "axes.edgecolor":       "#2f2f2f",
    "axes.facecolor":       "white",
    "axes.prop_cycle":      matplotlib.cycler(color=_PALETTE),

    # Ticks
    "xtick.labelsize":      10.5,
    "ytick.labelsize":      10.5,
    "xtick.major.width":    1.2,
    "ytick.major.width":    1.2,
    "xtick.minor.width":    0.8,
    "ytick.minor.width":    0.8,
    "xtick.direction":      "in",
    "ytick.direction":      "in",

    # Legend
    "legend.fontsize":      9.5,
    "legend.frameon":       True,
    "legend.framealpha":    0.93,
    "legend.edgecolor":     "#444444",
    "legend.fancybox":      True,
    "legend.shadow":        False,

    # Grid
    "grid.alpha":           0.2,
    "grid.linewidth":       0.6,

    # Figure
    "figure.facecolor":     "white",
})


# ── Major Iraqi & regional fault systems ──
FAULTS = {
    "Kirkuk Anticline":         (44.40, 35.47),
    "Zagros Suture":            (45.50, 35.20),
    "Hamrin Thrust":            (43.50, 34.50),
    "Mosul Fault":              (43.15, 36.34),
    "Sirnak-Cizre\n(Turkey)":  (42.50, 37.30),
    "Khanaqin Fault":           (45.39, 34.35),
    "Dukan Fault":              (44.95, 35.95),
    "Tigris Lineament":         (43.00, 33.50),
    "Erbil Fault":              (44.01, 36.19),
    "Shaqlawa Thrust":          (44.32, 36.64),
}

BAND_LABELS = ["0.5–1.5 Hz", "1.5–3.0 Hz", "3.0–6.0 Hz", "6.0–12.0 Hz"]
DEPTH_KEYS  = [0, 2, 5, 8, 12, 15]
FILL        = -9999.0


# ═══════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════
def load_data():
    print("Loading NetCDF...")
    nc = netcdf_file(NC_PATH, 'r', mmap=False)
    d = {}
    for vn in nc.variables:
        d[vn] = nc.variables[vn].data.copy()
    nc.close()
    for k in d:
        if d[k].dtype in [np.float32, np.float64]:
            d[k] = np.where(d[k] == FILL, np.nan, d[k])
    d["nx"] = len(d["longitude"])
    d["ny"] = len(d["latitude"])
    d["nz"] = len(d["depth"])
    d["nb"] = len(d["band"])
    print(f"  Grid: {d['nx']}×{d['ny']}×{d['nz']}, {d['nb']} bands")
    print(f"  Lon:   {d['longitude'][0]:.2f}–{d['longitude'][-1]:.2f}°E")
    print(f"  Lat:   {d['latitude'][0]:.2f}–{d['latitude'][-1]:.2f}°N")
    print(f"  Depth: {d['depth'][0]:.1f}–{d['depth'][-1]:.1f} km")
    return d


def savefig(fig, name):
    out = os.path.join(FIG_DIR, name)
    fig.savefig(out, dpi=DPI_SAVE, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"  [OK] {name}")


# ═══════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════
def _safe_twoslope(vmin, vmax, vcenter=0):
    """TwoSlopeNorm with full edge-case protection."""
    if not np.isfinite(vmin):
        vmin = -0.1
    if not np.isfinite(vmax):
        vmax = 0.1
    if vmin >= vcenter:
        vmin = vcenter - max(0.01, abs(vmax - vcenter))
    if vmax <= vcenter:
        vmax = vcenter + max(0.01, abs(vcenter - vmin))
    if vmin >= vmax:
        vmin, vmax = vcenter - 0.1, vcenter + 0.1
    return TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)


def get_sym_clim(arr, pct=3):
    """Symmetric color limits with zero guard."""
    f = arr[np.isfinite(arr)]
    if len(f) == 0:
        return -0.1, 0.1
    v = np.percentile(np.abs(f), 100 - pct)
    if v < 1e-12:
        v = max(abs(f.min()), abs(f.max()))
    if v < 1e-12:
        v = 0.1
    return -v, v


def _smart_colorbar(im, ax, label, fmt_sci=True):
    """Well-formatted colorbar."""
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(label, fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9.5)
    if fmt_sci:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 3))
        cbar.ax.yaxis.set_major_formatter(fmt)
    return cbar


def setup_map_ax(ax, extent, xlabel=True, ylabel=True):
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.tick_params(which="major", length=5, width=1.2, direction="in",
                   top=True, right=True)
    ax.tick_params(which="minor", length=2.5, width=0.7, direction="in",
                   top=True, right=True)
    ax.grid(True, which="major", alpha=0.2, lw=0.5, color="#444444")
    ax.grid(True, which="minor", alpha=0.08, lw=0.3, ls=":",
            color="#888888")
    if xlabel:
        ax.set_xlabel("Longitude (°E)")
    if ylabel:
        ax.set_ylabel("Latitude (°N)")


def _add_contours(ax, lons, lats, arr, n_levels=8, color="white"):
    """White contour overlay with labels."""
    finite = arr[np.isfinite(arr)]
    if finite.size < 20:
        return
    vmin = np.percentile(finite, 3)
    vmax = np.percentile(finite, 97)
    if abs(vmax - vmin) < 1e-12:
        return
    LON, LAT = np.meshgrid(lons, lats)
    levels = np.linspace(vmin, vmax, n_levels)
    try:
        cs = ax.contour(LON, LAT, arr, levels=levels, colors=color,
                        linewidths=0.4, alpha=0.45)
        ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.3g")
    except Exception:
        pass


def mark_faults(ax, subset=None):
    for name, (lo, la) in FAULTS.items():
        if subset and name not in subset:
            continue
        ax.plot(lo, la, "^", color="#1a1a1a", ms=5.5, mew=0.7, zorder=7)
        ax.text(
            lo + 0.15, la + 0.1, name,
            fontsize=6.5, fontweight="bold", color="#1a1a1a",
            path_effects=STROKE_THIN, zorder=8,
        )


def mark_kirkuk(ax):
    ax.plot(
        44.39, 35.47, "*", color="#ffd700", ms=13,
        mec="#1a1a1a", mew=0.8, zorder=9,
    )


def _clamp_idx(arr, target):
    """Return index nearest to target, clamped to valid range."""
    return int(np.clip(np.argmin(np.abs(arr - target)), 0, len(arr) - 1))


# ═══════════════════════════════════════════════════════
#  FIGURE 1: Qi DEPTH SLICES (6 panels, Band 2)
# ═══════════════════════════════════════════════════════
def fig_qi_depth_slices(D):
    band = 2
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    extent = [40.5, 47.5, 31, 38.5]
    vmin, vmax = get_sym_clim(D["Qi_inv"][band])

    for i, z in enumerate(DEPTH_KEYS):
        ax = axes.flat[i]
        iz = _clamp_idx(D["depth"], z)
        arr = D["Qi_inv"][band, iz]
        norm = _safe_twoslope(vmin, vmax)
        im = ax.pcolormesh(
            D["longitude"], D["latitude"], arr,
            cmap=CMAP_QI, norm=norm, shading="gouraud", rasterized=True,
        )
        _add_contours(ax, D["longitude"], D["latitude"], arr)
        setup_map_ax(ax, extent)
        mark_faults(ax)
        mark_kirkuk(ax)
        ax.set_title(f"Depth = {D['depth'][iz]:.0f} km", fontsize=12)
        label = "$Q_i^{-1}$" if i in [2, 5] else ""
        _smart_colorbar(im, ax, label)

    fig.suptitle(
        "Figure 1. Intrinsic Attenuation ($Q_i^{-1}$) — 3.0–6.0 Hz\n"
        "Depth slices showing lateral variation and "
        "fault-correlated anomalies",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, "Fig01_Qi_depth_slices.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 2: Qsct DEPTH SLICES
# ═══════════════════════════════════════════════════════
def fig_qsct_depth_slices(D):
    band = 2
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    extent = [40.5, 47.5, 31, 38.5]
    vmin, vmax = get_sym_clim(D["Qsct_inv"][band])

    for i, z in enumerate(DEPTH_KEYS):
        ax = axes.flat[i]
        iz = _clamp_idx(D["depth"], z)
        arr = D["Qsct_inv"][band, iz]
        norm = _safe_twoslope(vmin, vmax)
        im = ax.pcolormesh(
            D["longitude"], D["latitude"], arr,
            cmap=CMAP_QSCT, norm=norm, shading="gouraud", rasterized=True,
        )
        _add_contours(ax, D["longitude"], D["latitude"], arr)
        setup_map_ax(ax, extent)
        mark_faults(ax)
        mark_kirkuk(ax)
        ax.set_title(f"Depth = {D['depth'][iz]:.0f} km")
        label = "$Q_{sc}^{-1}$" if i in [2, 5] else ""
        _smart_colorbar(im, ax, label)

    fig.suptitle(
        "Figure 2. Scattering Attenuation ($Q_{sc}^{-1}$) — 3.0–6.0 Hz\n"
        "Heterogeneity structure related to fault zone complexity",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, "Fig02_Qsct_depth_slices.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 3: Qt TOTAL ATTENUATION
# ═══════════════════════════════════════════════════════
def fig_qt_depth_slices(D):
    band = 2
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    extent = [40.5, 47.5, 31, 38.5]
    vmin, vmax = get_sym_clim(D["Qt_inv"][band])

    for i, z in enumerate(DEPTH_KEYS):
        ax = axes.flat[i]
        iz = _clamp_idx(D["depth"], z)
        arr = D["Qt_inv"][band, iz]
        norm = _safe_twoslope(vmin, vmax)
        im = ax.pcolormesh(
            D["longitude"], D["latitude"], arr,
            cmap=CMAP_QT, norm=norm, shading="gouraud", rasterized=True,
        )
        _add_contours(ax, D["longitude"], D["latitude"], arr)
        setup_map_ax(ax, extent)
        mark_faults(ax)
        mark_kirkuk(ax)
        ax.set_title(f"Depth = {D['depth'][iz]:.0f} km")
        label = "$Q_t^{-1}$" if i in [2, 5] else ""
        _smart_colorbar(im, ax, label)

    fig.suptitle(
        "Figure 3. Total Attenuation "
        "($Q_t^{-1} = Q_i^{-1} + Q_{sc}^{-1}$) — 3.0–6.0 Hz\n"
        "Combined energy loss along fault corridors",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, "Fig03_Qt_depth_slices.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 4: FREQUENCY-DEPENDENT Qi (4 bands at z=5 km)
# ═══════════════════════════════════════════════════════
def fig_freq_dependent_qi(D):
    iz = _clamp_idx(D["depth"], 5.0)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    extent = [40.5, 47.5, 31, 38.5]

    for b in range(min(4, D["nb"])):
        ax = axes[b]
        arr = D["Qi_inv"][b, iz]
        vmin, vmax = get_sym_clim(arr)
        norm = _safe_twoslope(vmin, vmax)
        im = ax.pcolormesh(
            D["longitude"], D["latitude"], arr,
            cmap=CMAP_QI, norm=norm, shading="gouraud", rasterized=True,
        )
        _add_contours(ax, D["longitude"], D["latitude"], arr, n_levels=6)
        setup_map_ax(ax, extent, ylabel=(b == 0))
        mark_faults(ax)
        mark_kirkuk(ax)
        ax.set_title(f"{BAND_LABELS[b]}", fontsize=12)
        _smart_colorbar(im, ax, "$Q_i^{-1}$" if b == 3 else "")

    fig.suptitle(
        "Figure 4. Frequency-Dependent $Q_i^{-1}$ at 5 km Depth\n"
        "Higher frequencies resolve finer fault structures",
        fontsize=14, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    savefig(fig, "Fig04_freq_dependent_Qi.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 5: Qsct/Qi RATIO MAP
# ═══════════════════════════════════════════════════════
def fig_ratio_maps(D):
    band = 2
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    extent = [40.5, 47.5, 31, 38.5]

    for i, z in enumerate(DEPTH_KEYS):
        ax = axes.flat[i]
        iz = _clamp_idx(D["depth"], z)
        arr = D["Qsct_Qi_ratio"][band, iz].copy()
        arr = np.clip(arr, -3, 3)
        norm = _safe_twoslope(0, 2.5, vcenter=1)
        im = ax.pcolormesh(
            D["longitude"], D["latitude"], arr,
            cmap=CMAP_RATIO, norm=norm, shading="gouraud", rasterized=True,
        )
        _add_contours(ax, D["longitude"], D["latitude"], arr, n_levels=6)
        setup_map_ax(ax, extent)
        mark_faults(ax)
        mark_kirkuk(ax)
        ax.set_title(f"Depth = {D['depth'][iz]:.0f} km")
        label = "$Q_{sc}^{-1}/Q_i^{-1}$" if i in [2, 5] else ""
        _smart_colorbar(im, ax, label, fmt_sci=False)

    fig.suptitle(
        "Figure 5. Scattering-to-Intrinsic Ratio "
        "($Q_{sc}^{-1}/Q_i^{-1}$) — 3.0–6.0 Hz\n"
        "Ratio > 1 (warm tones): scattering-dominant = "
        "fractured/faulted rock",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, "Fig05_Qsct_Qi_ratio.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 6: E-W & N-S CROSS-SECTIONS (Qi + Qsct)
# ═══════════════════════════════════════════════════════
def extract_section(D, band, direction, **kw):
    if direction == "EW":
        iy = _clamp_idx(D["latitude"], kw["lat"])
        return (D["longitude"], D["depth"],
                D["Qi_inv"][band, :, iy, :],
                D["Qsct_inv"][band, :, iy, :])
    elif direction == "NS":
        ix = _clamp_idx(D["longitude"], kw["lon"])
        return (D["latitude"], D["depth"],
                D["Qi_inv"][band, :, :, ix],
                D["Qsct_inv"][band, :, :, ix])


def _plot_section_pair(fig, gs_left_right, x, z, qi_s, qs_s,
                       title_qi, title_ns, color, xlims,
                       ref_val=None, xlabel=None):
    """Plot a Qi + Qsct section pair in one gridspec row."""
    gs_left, gs_right = gs_left_right
    # Qi
    ax = fig.add_subplot(gs_left)
    v1, v2 = get_sym_clim(qi_s)
    norm = _safe_twoslope(v1, v2)
    im = ax.pcolormesh(
        x, z, qi_s, cmap=CMAP_QI, norm=norm,
        shading="gouraud", rasterized=True,
    )
    ax.invert_yaxis()
    if ref_val is not None:
        ax.axvline(ref_val, color="#ffd700", lw=2, ls="--", alpha=0.8)
    ax.set_ylabel("Depth (km)")
    ax.set_title(title_qi, color=color, fontsize=12, fontweight="bold")
    ax.set_xlim(xlims)
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, alpha=0.12)
    _smart_colorbar(im, ax, "$Q_i^{-1}$")
    if xlabel:
        ax.set_xlabel(xlabel)

    # Qsct
    ax2 = fig.add_subplot(gs_right)
    v1, v2 = get_sym_clim(qs_s)
    norm2 = _safe_twoslope(v1, v2)
    im2 = ax2.pcolormesh(
        x, z, qs_s, cmap=CMAP_QSCT, norm=norm2,
        shading="gouraud", rasterized=True,
    )
    ax2.invert_yaxis()
    if ref_val is not None:
        ax2.axvline(ref_val, color="#ffd700", lw=2, ls="--", alpha=0.8)
    ax2.set_title(title_ns, color=color, fontsize=12, fontweight="bold")
    ax2.set_xlim(xlims)
    ax2.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax2.grid(True, alpha=0.12)
    _smart_colorbar(im2, ax2, "$Q_{sc}^{-1}$")
    if xlabel:
        ax2.set_xlabel(xlabel)


def fig_cross_sections(D):
    band = 2
    fig = plt.figure(figsize=(19, 17))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.26)

    # A-A': E-W through Kirkuk
    x, z, qi_s, qs_s = extract_section(D, band, "EW", lat=35.45)
    _plot_section_pair(
        fig, (gs[0, 0], gs[0, 1]), x, z, qi_s, qs_s,
        "A–A': E–W $Q_i^{-1}$ at 35.45°N",
        "A–A': E–W $Q_{sc}^{-1}$ at 35.45°N",
        "#e63946", (41, 47), ref_val=44.39,
    )

    # B-B': N-S through Kirkuk
    x2, z2, qi_n, qs_n = extract_section(D, band, "NS", lon=44.44)
    _plot_section_pair(
        fig, (gs[1, 0], gs[1, 1]), x2, z2, qi_n, qs_n,
        "B–B': N–S $Q_i^{-1}$ at 44.44°E",
        "B–B': N–S $Q_{sc}^{-1}$ at 44.44°E",
        "#2a9d8f", (32, 38), ref_val=35.47,
    )

    # D-D': E-W through Hamrin
    x3, z3, qi_h, qs_h = extract_section(D, band, "EW", lat=34.5)
    _plot_section_pair(
        fig, (gs[2, 0], gs[2, 1]), x3, z3, qi_h, qs_h,
        "D–D': E–W $Q_i^{-1}$ at 34.5°N (Hamrin–Zagros)",
        "D–D': E–W $Q_{sc}^{-1}$ at 34.5°N (Hamrin–Zagros)",
        "#a855f7", (41, 47),
    )

    # N-S through Mosul-Hamrin
    x4, z4, qi_m, qs_m = extract_section(D, band, "NS", lon=43.2)
    _plot_section_pair(
        fig, (gs[3, 0], gs[3, 1]), x4, z4, qi_m, qs_m,
        "N–S $Q_i^{-1}$ at 43.2°E (Mosul corridor)",
        "N–S $Q_{sc}^{-1}$ at 43.2°E (Mosul corridor)",
        "#264653", (32, 38),
        xlabel="Latitude (°N)",
    )

    fig.suptitle(
        "Figure 6. Vertical Cross-Sections — "
        "$Q_i^{-1}$ and $Q_{sc}^{-1}$ (3.0–6.0 Hz)\n"
        "Fault zones visible as lateral attenuation "
        "contrasts extending to depth",
        fontsize=14, fontweight="bold", y=1.01,
    )
    savefig(fig, "Fig06_cross_sections.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 7: UNCERTAINTY / RELIABILITY MAPS
# ═══════════════════════════════════════════════════════
def fig_uncertainty(D):
    band = 2
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
    extent = [40.5, 47.5, 31, 38.5]

    for i, z in enumerate(DEPTH_KEYS):
        ax = axes.flat[i]
        iz = _clamp_idx(D["depth"], z)
        arr = D["Qi_inv_std"][band, iz]
        im = ax.pcolormesh(
            D["longitude"], D["latitude"], arr,
            cmap=CMAP_STD, shading="gouraud", vmin=0, rasterized=True,
        )
        _add_contours(ax, D["longitude"], D["latitude"], arr, n_levels=6,
                      color="#cccccc")
        setup_map_ax(ax, extent)
        mark_faults(ax)
        mark_kirkuk(ax)
        ax.set_title(f"Depth = {D['depth'][iz]:.0f} km")
        label = "$\\sigma(Q_i^{-1})$" if i in [2, 5] else ""
        _smart_colorbar(im, ax, label)

    fig.suptitle(
        "Figure 7. Ensemble Uncertainty $\\sigma(Q_i^{-1})$ — "
        "3.0–6.0 Hz\n"
        "Low uncertainty = well-resolved; "
        "high = poor ray coverage",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, "Fig07_Qi_uncertainty.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 8: DEPTH PROFILES AT FAULT LOCATIONS
# ═══════════════════════════════════════════════════════
def fig_depth_profiles(D):
    band = 2
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.5))
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(FAULTS)))

    for fi, (name, (lo, la)) in enumerate(FAULTS.items()):
        ix = _clamp_idx(D["longitude"], lo)
        iy = _clamp_idx(D["latitude"], la)
        qi_prof = D["Qi_inv"][band, :, iy, ix]
        qs_prof = D["Qsct_inv"][band, :, iy, ix]
        qt_prof = D["Qt_inv"][band, :, iy, ix]
        c = colors[fi]
        label = name.replace("\n", " ")

        axes[0].plot(qi_prof, D["depth"], "o-", color=c, ms=4,
                     lw=1.8, label=label)
        axes[1].plot(qs_prof, D["depth"], "s-", color=c, ms=4,
                     lw=1.8, label=label)
        axes[2].plot(qt_prof, D["depth"], "^-", color=c, ms=4,
                     lw=1.8, label=label)

    titles = ["$Q_i^{-1}$", "$Q_{sc}^{-1}$", "$Q_t^{-1}$"]
    for ax, title in zip(axes, titles):
        ax.invert_yaxis()
        ax.axvline(0, color="#888888", lw=0.6, ls="--")
        ax.set_ylabel("Depth (km)")
        ax.set_xlabel(title, fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=7, loc="lower right", framealpha=0.92)

    fig.suptitle(
        "Figure 8. Depth Profiles at Major Fault Locations "
        "(3.0–6.0 Hz)\n"
        "Vertical extent and depth variation of "
        "fault-related anomalies",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, "Fig08_depth_profiles.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 9: Qi vs FREQUENCY AT EACH FAULT
# ═══════════════════════════════════════════════════════
def fig_qi_vs_freq(D):
    iz = _clamp_idx(D["depth"], 5.0)
    fc = D["band_freq_center"]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(FAULTS)))

    for fi, (name, (lo, la)) in enumerate(FAULTS.items()):
        ix = _clamp_idx(D["longitude"], lo)
        iy = _clamp_idx(D["latitude"], la)
        nb = min(4, D["nb"])
        vals = [D["Qi_inv"][b, iz, iy, ix] for b in range(nb)]
        stds = [D["Qi_inv_std"][b, iz, iy, ix] for b in range(nb)]
        label = name.replace("\n", " ")
        ax.errorbar(
            fc[:nb], vals, yerr=stds, fmt="o-", color=colors[fi],
            lw=2.2, ms=7, capsize=5, capthick=1.5, label=label,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel("$Q_i^{-1}$", fontsize=14, fontweight="bold")
    ax.set_title(
        "Figure 9. Frequency-Dependent $Q_i^{-1}$ at Fault "
        "Locations (z = 5 km)\n"
        "Error bars: ensemble uncertainty",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(0, color="#888888", lw=0.6, ls="--")
    ax.legend(fontsize=8.5, ncol=2, framealpha=0.92)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    savefig(fig, "Fig09_Qi_vs_frequency.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 10: SNR RELIABILITY MAP
# ═══════════════════════════════════════════════════════
def fig_snr_map(D):
    band = 2
    iz = _clamp_idx(D["depth"], 5.0)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    extent = [40.5, 47.5, 31, 38.5]

    for i, (vn, title) in enumerate([
        ("Qi_SNR",   "$Q_i$ SNR"),
        ("Qsct_SNR", "$Q_{sc}$ SNR"),
    ]):
        ax = axes[i]
        arr = np.clip(D[vn][band, iz], 0, 5)
        im = ax.pcolormesh(
            D["longitude"], D["latitude"], arr,
            cmap=CMAP_SNR, shading="gouraud", vmin=0, vmax=5,
            rasterized=True,
        )
        _add_contours(ax, D["longitude"], D["latitude"], arr, n_levels=5,
                      color="#333333")
        setup_map_ax(ax, extent)
        mark_faults(ax)
        mark_kirkuk(ax)
        ax.set_title(f"{title} at 5 km depth", fontsize=13)
        _smart_colorbar(im, ax, "|mean|/σ", fmt_sci=False)

    fig.suptitle(
        "Figure 10. Inversion Reliability "
        "(SNR = |mean|/$\\sigma$) — 3.0–6.0 Hz\n"
        "SNR > 2: well-resolved; SNR < 0.5: poorly constrained",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    savefig(fig, "Fig10_SNR_reliability.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 11: CONVERGENCE CURVES
# ═══════════════════════════════════════════════════════
def fig_convergence(D):
    if "loss_total" not in D:
        print("  [SKIP] Fig11 — loss_total not found in NetCDF")
        return

    loss = D["loss_total"]
    if loss.ndim < 1 or loss.shape[0] == 0:
        print("  [SKIP] Fig11 — loss_total is empty")
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_members = loss.shape[0] if loss.ndim == 2 else 1
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, n_members))

    if loss.ndim == 2:
        for m in range(n_members):
            vals = loss[m]
            valid = vals[np.isfinite(vals)]
            if len(valid) == 0:
                continue
            label = f"Member {m + 1}" if m < 4 else None
            ax.semilogy(
                range(len(valid)), valid,
                alpha=0.6, lw=1.5, color=colors[m], label=label,
            )
    else:
        valid = loss[np.isfinite(loss)]
        ax.semilogy(range(len(valid)), valid, lw=2, color=_PALETTE[0])

    ax.set_xlabel("Iteration", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Loss (log scale)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Figure 11. Inversion Convergence — All Ensemble Members",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.92)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    savefig(fig, "Fig11_convergence.png")


# ═══════════════════════════════════════════════════════
#  FIGURE 12: Qi-Qsct-Qt COMPARISON AT 5 km
# ═══════════════════════════════════════════════════════
def fig_qi_qsct_comparison(D):
    band = 2
    iz = _clamp_idx(D["depth"], 5.0)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    extent = [40.5, 47.5, 31, 38.5]

    configs = [
        ("Qi_inv",   CMAP_QI,   "$Q_i^{-1}$"),
        ("Qsct_inv", CMAP_QSCT, "$Q_{sc}^{-1}$"),
        ("Qt_inv",   CMAP_QT,   "$Q_t^{-1}$"),
    ]

    for i, (vn, cmap, label) in enumerate(configs):
        ax = axes[i]
        arr = D[vn][band, iz]
        v1, v2 = get_sym_clim(arr)
        norm = _safe_twoslope(v1, v2)
        im = ax.pcolormesh(
            D["longitude"], D["latitude"], arr,
            cmap=cmap, norm=norm, shading="gouraud", rasterized=True,
        )
        _add_contours(ax, D["longitude"], D["latitude"], arr, n_levels=8)
        setup_map_ax(ax, extent, ylabel=(i == 0))
        mark_faults(ax)
        mark_kirkuk(ax)
        ax.set_title(f"{label} at 5 km", fontsize=13, fontweight="bold")
        _smart_colorbar(im, ax, label)

    fig.suptitle(
        "Figure 12. Comparison: $Q_i^{-1}$, $Q_{sc}^{-1}$, "
        "$Q_t^{-1}$ at 5 km — 3.0–6.0 Hz\n"
        "Separation of intrinsic and scattering contributions "
        "along fault zones",
        fontsize=13, fontweight="bold", y=1.03,
    )
    plt.tight_layout()
    savefig(fig, "Fig12_Qi_Qsct_Qt_comparison.png")


# ═══════════════════════════════════════════════════════
#  TABLE 1: FAULT-BY-FAULT ATTENUATION SUMMARY
# ═══════════════════════════════════════════════════════
def table_fault_summary(D):
    rows = []
    nb = min(4, D["nb"])
    for name, (lo, la) in FAULTS.items():
        name_clean = name.replace("\n", " ")
        ix = _clamp_idx(D["longitude"], lo)
        iy = _clamp_idx(D["latitude"], la)
        for b in range(nb):
            for z in [0, 5, 10, 15]:
                iz = _clamp_idx(D["depth"], z)
                qi  = D["Qi_inv"][b, iz, iy, ix]
                qs  = D["Qsct_inv"][b, iz, iy, ix]
                qt  = D["Qt_inv"][b, iz, iy, ix]
                qi_e = D["Qi_inv_std"][b, iz, iy, ix]
                qs_e = D["Qsct_inv_std"][b, iz, iy, ix]
                snr = D["Qi_SNR"][b, iz, iy, ix]
                rat = D["Qsct_Qi_ratio"][b, iz, iy, ix]

                def _fmt(v, d=5):
                    return f"{v:.{d}f}" if np.isfinite(v) else "NaN"

                rows.append({
                    "Fault":     name_clean,
                    "Lon":       lo,
                    "Lat":       la,
                    "Band":      BAND_LABELS[b],
                    "Depth_km":  z,
                    "Qi_inv":    _fmt(qi),
                    "Qi_std":    _fmt(qi_e),
                    "Qsct_inv":  _fmt(qs),
                    "Qsct_std":  _fmt(qs_e),
                    "Qt_inv":    _fmt(qt),
                    "Ratio":     (_fmt(rat, 3)
                                 if np.isfinite(rat) and abs(rat) < 100
                                 else "NaN"),
                    "SNR":       _fmt(snr, 2),
                })

    df = pd.DataFrame(rows)
    out = os.path.join(TBL_DIR, "Table1_fault_attenuation_summary.csv")
    df.to_csv(out, index=False)
    print(f"  [OK] {out} ({len(df)} rows)")

    # Compact version (Band 2, z=5 km)
    compact = df[
        (df.Band == "3.0–6.0 Hz") & (df.Depth_km == 5)
    ].copy()
    compact = compact[[
        "Fault", "Lon", "Lat", "Qi_inv", "Qi_std",
        "Qsct_inv", "Qt_inv", "Ratio", "SNR",
    ]]
    out2 = os.path.join(TBL_DIR, "Table1_compact.csv")
    compact.to_csv(out2, index=False)
    print(f"  [OK] {out2}")
    return df


# ═══════════════════════════════════════════════════════
#  TABLE 2: DEPTH-AVERAGED REGIONAL STATISTICS
# ═══════════════════════════════════════════════════════
def table_regional_stats(D):
    band = 2
    regions = {
        "Kirkuk Embayment":   ((43.8, 45.0), (35.0, 36.0)),
        "Zagros Foothills":   ((45.0, 46.5), (35.0, 36.5)),
        "Hamrin-Makhul":      ((43.0, 44.5), (34.0, 35.5)),
        "Mesopotamian Plain": ((43.0, 45.5), (31.5, 34.0)),
        "N. Iraq-Turkey":     ((42.5, 44.5), (36.5, 38.0)),
        "Entire Study Area":  ((40.5, 47.5), (31.0, 38.5)),
    }

    rows = []
    for rname, ((lo1, lo2), (la1, la2)) in regions.items():
        mask_x = (D["longitude"] >= lo1) & (D["longitude"] <= lo2)
        mask_y = (D["latitude"] >= la1) & (D["latitude"] <= la2)
        ix = np.where(mask_x)[0]
        iy = np.where(mask_y)[0]

        for z in [0, 5, 10, 15]:
            iz = _clamp_idx(D["depth"], z)
            sub_qi  = D["Qi_inv"][band, iz][np.ix_(iy, ix)]
            sub_qs  = D["Qsct_inv"][band, iz][np.ix_(iy, ix)]
            sub_qt  = D["Qt_inv"][band, iz][np.ix_(iy, ix)]
            sub_std = D["Qi_inv_std"][band, iz][np.ix_(iy, ix)]

            def _stats(a):
                f = a[np.isfinite(a)]
                if len(f) == 0:
                    return "NaN", "NaN", 0
                return f"{f.mean():.5f}", f"{f.std():.5f}", len(f)

            qi_m, qi_s, nv = _stats(sub_qi)
            qs_m, _, _ = _stats(sub_qs)
            qt_m, _, _ = _stats(sub_qt)
            unc_m, _, _ = _stats(sub_std)

            rows.append({
                "Region":           rname,
                "Depth_km":         z,
                "Qi_mean":          qi_m,
                "Qi_std":           qi_s,
                "Qsct_mean":        qs_m,
                "Qt_mean":          qt_m,
                "Uncertainty_mean": unc_m,
                "N_voxels":         nv,
            })

    df = pd.DataFrame(rows)
    out = os.path.join(TBL_DIR, "Table2_regional_statistics.csv")
    df.to_csv(out, index=False)
    print(f"  [OK] {out}")
    return df


# ═══════════════════════════════════════════════════════
#  TABLE 3: FREQUENCY-DEPENDENT ATTENUATION
# ═══════════════════════════════════════════════════════
def table_freq_dependence(D):
    iz = _clamp_idx(D["depth"], 5.0)
    nb = min(4, D["nb"])
    rows = []
    for name, (lo, la) in FAULTS.items():
        name_clean = name.replace("\n", " ")
        ix = _clamp_idx(D["longitude"], lo)
        iy = _clamp_idx(D["latitude"], la)
        for b in range(nb):
            qi = D["Qi_inv"][b, iz, iy, ix]
            qs = D["Qsct_inv"][b, iz, iy, ix]
            rows.append({
                "Fault":    name_clean,
                "Band":     BAND_LABELS[b],
                "Freq_Hz":  f"{D['band_freq_center'][b]:.2f}",
                "Qi_inv":   f"{qi:.5f}" if np.isfinite(qi) else "NaN",
                "Qsct_inv": f"{qs:.5f}" if np.isfinite(qs) else "NaN",
            })

    df = pd.DataFrame(rows)
    out = os.path.join(TBL_DIR, "Table3_freq_dependence.csv")
    df.to_csv(out, index=False)
    print(f"  [OK] {out}")


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  PUBLICATION FIGURE SUITE v4.0")
    print("  Iraq Attenuation Tomography — Fault Segmentation")
    print("=" * 65)
    print(f"  Output:  {FIG_DIR}")
    print(f"  DPI:     {DPI_SAVE}")
    print()

    D = load_data()

    print("\n--- FIGURES ---")
    fig_qi_depth_slices(D)          # Fig 1
    fig_qsct_depth_slices(D)        # Fig 2
    fig_qt_depth_slices(D)          # Fig 3
    fig_freq_dependent_qi(D)        # Fig 4
    fig_ratio_maps(D)               # Fig 5
    fig_cross_sections(D)           # Fig 6
    fig_uncertainty(D)              # Fig 7
    fig_depth_profiles(D)           # Fig 8
    fig_qi_vs_freq(D)               # Fig 9
    fig_snr_map(D)                  # Fig 10
    fig_convergence(D)              # Fig 11
    fig_qi_qsct_comparison(D)       # Fig 12

    print("\n--- TABLES ---")
    table_fault_summary(D)          # Table 1
    table_regional_stats(D)         # Table 2
    table_freq_dependence(D)        # Table 3

    print(f"\n{'=' * 65}")
    print(f"  All outputs: {FIG_DIR}")
    print(f"  12 figures + 3 tables ready for submission")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()


# In[ ]:




