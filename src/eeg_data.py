from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import torch
import mne
from scipy.signal import welch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "files"

# EEG configuration
CHANNELS = ["C3", "Cz", "C4"]
BANDS = [(8, 12), (13, 30)]
EPOCH_TMIN, EPOCH_TMAX = 0.0, 2.0
SUPPORT_RUNS = [4, 8]
QUERY_RUNS = [12]
DEFAULT_K_SUPPORT = 8
DEFAULT_K_QUERY = 20


def bandpower_features(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    feats = []
    for ch in range(epoch.shape[0]):
        f, pxx = welch(epoch[ch], fs=sfreq, nperseg=min(256, epoch.shape[1]))
        for lo, hi in BANDS:
            mask = (f >= lo) & (f <= hi)
            feats.append(np.log(np.trapz(pxx[mask], f[mask]) + 1e-12))
    return np.array(feats, dtype=np.float32)

def _normalize_event_id(event_id: dict) -> dict:
    return {str(k).strip(): v for k, v in event_id.items()}

def load_subject_runs(subject_id: int, runs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    subj = f"S{subject_id:03d}"
    edf_files = []
    for r in runs:
        f = DATA_ROOT / subj / f"{subj}R{r:02d}.edf"
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}")
        edf_files.append(str(f))

    raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in edf_files]
    raw = mne.concatenate_raws(raws)

    raw.rename_channels(lambda s: s.strip('.'))
    raw.set_montage("standard_1020", on_missing="ignore")

    present = [c for c in CHANNELS if c in raw.ch_names]
    if len(present) < 2:
        raise RuntimeError(f"{subj}: not enough of {CHANNELS} present (found {present})")

    raw.pick_channels(present)
    raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_id = _normalize_event_id(event_id)

    if "T1" not in event_id or "T2" not in event_id:
        raise RuntimeError(f"{subj}: expected T1/T2 not found. Found {list(event_id.keys())}")

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"T1": event_id["T1"], "T2": event_id["T2"]},
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )
    data = epochs.get_data()  
    labels = epochs.events[:, 2]
    y = (labels == event_id["T2"]).astype(np.int64)

    X = np.stack([bandpower_features(ep, epochs.info["sfreq"]) for ep in data], axis=0)
    return X, y

def _balanced_pick(X: np.ndarray, y: np.ndarray, k: int, rng: np.random.Generator):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    need = k // 2

    if len(idx0) == 0 or len(idx1) == 0:
        return None

    rep0 = len(idx0) < need
    rep1 = len(idx1) < need

    a0 = rng.choice(idx0, size=need, replace=rep0)
    a1 = rng.choice(idx1, size=need, replace=rep1)

    idx = np.concatenate([a0, a1])
    rng.shuffle(idx)
    return X[idx], y[idx]

def sample_within_subject_task(subject_id: int, k_support: int = DEFAULT_K_SUPPORT,
    k_query: int = DEFAULT_K_QUERY, seed: Optional[int] = None, device: Optional[torch.device] = None,):
    if device is None:
        device = torch.device("cpu")

    rng = np.random.default_rng(seed)

    Xs, ys = load_subject_runs(subject_id, SUPPORT_RUNS)
    Xq, yq = load_subject_runs(subject_id, QUERY_RUNS)

    s = _balanced_pick(Xs, ys, k_support, rng)
    q = _balanced_pick(Xq, yq, k_query, rng)
    if s is None or q is None:
        return None

    xs, ys_ = s
    xq, yq_ = q

    xs = torch.tensor(xs, dtype=torch.float32, device=device)
    ys_ = torch.tensor(ys_, dtype=torch.long, device=device)
    xq = torch.tensor(xq, dtype=torch.float32, device=device)
    yq_ = torch.tensor(yq_, dtype=torch.long, device=device)

    return xs, ys_, xq, yq_
