
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple

# =============================================================================
# 1. Hardcoded Parameters from 7th Place Solution
# =============================================================================

TT_PER_LAB_NN = {'AdaptableSnail': {'approach': 0.78,
  'attack': 0.59,
  'avoid': 0.63,
  'chase': 0.61,
  'chaseattack': 0.6900000000000001,
  'rear': 0.7000000000000001,
  'submit': 0.22000000000000008},
 'BoisterousParrot': {'shepherd': 0.7100000000000001},
 'CRIM13': {'approach': 0.79,
  'attack': 0.59,
  'disengage': 0.7200000000000001,
  'mount': 0.48000000000000004,
  'rear': 0.58,
  'selfgroom': 0.67,
  'sniff': 0.6900000000000001},
 'CalMS21_supplemental': {'approach': 0.89,
  'attack': 0.89,
  'attemptmount': 0.6,
  'dominancemount': 0.79,
  'intromit': 0.7300000000000001,
  'mount': 0.8300000000000001,
  'sniff': 0.7300000000000001,
  'sniffbody': 0.76,
  'sniffface': 0.88,
  'sniffgenital': 0.68},
 'CalMS21_task1': {'approach': 0.9,
  'attack': 0.88,
  'genitalgroom': 0.7100000000000001,
  'intromit': 0.7100000000000001,
  'mount': 0.8400000000000001,
  'sniff': 0.7000000000000001,
  'sniffbody': 0.6900000000000001,
  'sniffface': 0.87,
  'sniffgenital': 0.85},
 'CalMS21_task2': {'attack': 0.87,
  'mount': 0.12000000000000001,
  'sniff': 0.63},
 'CautiousGiraffe': {'chase': 0.3200000000000001,
  'escape': 0.8,
  'reciprocalsniff': 0.8500000000000001,
  'sniff': 0.64,
  'sniffbody': 0.54,
  'sniffgenital': 0.7100000000000001},
 'DeliriousFly': {'attack': 0.61,
  'dominance': 0.49000000000000005,
  'sniff': 0.51},
 'ElegantMink': {'allogroom': 0.35000000000000014,
  'attack': 0.49000000000000005,
  'attemptmount': 0.6900000000000001,
  'ejaculate': 0.18000000000000005,
  'intromit': 0.57,
  'mount': 0.62,
  'sniff': 0.38000000000000006},
 'GroovyShrew': {'approach': 0.86,
  'attemptmount': 0.34000000000000014,
  'climb': 0.30000000000000004,
  'defend': 0.5,
  'dig': 0.67,
  'escape': 0.7100000000000001,
  'rear': 0.76,
  'rest': 0.61,
  'run': 0.7200000000000001,
  'selfgroom': 0.76,
  'sniff': 0.62,
  'sniffgenital': 0.7200000000000001},
 'InvincibleJellyfish': {'allogroom': 0.35000000000000014,
  'attack': 0.7200000000000001,
  'dig': 0.65,
  'dominancegroom': 0.54,
  'escape': 0.35000000000000014,
  'selfgroom': 0.49000000000000005,
  'sniff': 0.67,
  'sniffgenital': 0.68},
 'JovialSwallow': {'attack': 0.7200000000000001,
  'chase': 0.2800000000000001,
  'sniff': 0.6900000000000001},
 'LyricalHare': {'approach': 0.25000000000000006,
  'attack': 0.77,
  'defend': 0.62,
  'escape': 0.8300000000000001,
  'freeze': 0.2,
  'rear': 0.7000000000000001,
  'sniff': 0.61},
 'NiftyGoldfinch': {'approach': 0.87,
  'attack': 0.7400000000000001,
  'biteobject': 0.37000000000000005,
  'chase': 0.86,
  'climb': 0.67,
  'defend': 0.66,
  'dig': 0.76,
  'escape': 0.87,
  'exploreobject': 0.57,
  'flinch': 0.64,
  'follow': 0.7300000000000001,
  'rear': 0.63,
  'selfgroom': 0.7100000000000001,
  'sniff': 0.58,
  'sniffface': 0.88,
  'sniffgenital': 0.4100000000000001,
  'tussle': 0.45},
 'PleasantMeerkat': {'attack': 0.59,
  'chase': 0.8200000000000001,
  'escape': 0.6900000000000001,
  'follow': 0.67},
 'ReflectiveManatee': {'attack': 0.86, 'sniff': 0.68},
 'SparklingTapir': {'attack': 0.86,
  'defend': 0.88,
  'escape': 0.9,
  'mount': 0.8300000000000001},
 'TranquilPanther': {'intromit': 0.63,
  'mount': 0.7500000000000001,
  'rear': 0.5599999999999999,
  'selfgroom': 0.5,
  'sniff': 0.62,
  'sniffgenital': 0.65},
 'UppityFerret': {'huddle': 0.65,
  'reciprocalsniff': 0.81,
  'sniffgenital': 0.7200000000000001},
 'unknown': {'allogroom': 0.34000000000000014, 'approach': 0.9400000000000001, 'attack': 0.77, 'attemptmount': 0.53, 'avoid': 0.75, 'biteobject': 0.55, 'chase': 0.8400000000000001, 'chaseattack': 0.76, 'climb': 0.48000000000000004, 'defend': 0.59, 'dig': 0.8, 'disengage': 0.8400000000000001, 'dominance': 0.63, 'dominancegroom': 0.66, 'dominancemount': 0.93, 'ejaculate': 0.35000000000000003, 'escape': 0.8300000000000001, 'exploreobject': 0.7400000000000001, 'flinch': 0.78, 'follow': 0.89, 'freeze': 0.30000000000000004, 'genitalgroom': 0.59, 'huddle': 0.6, 'intromit': 0.49000000000000005, 'mount': 0.7500000000000001, 'rear': 0.64, 'reciprocalsniff': 0.87, 'rest': 0.66, 'run': 0.89, 'selfgroom': 0.8, 'shepherd': 0.88, 'sniff': 0.61, 'sniffbody': 0.63, 'sniffface': 0.91, 'sniffgenital': 0.68, 'submit': 0.37000000000000005, 'tussle': 0.5499999999999999}}

TIE_CONFIG_V2 = {'InvincibleJellyfish': {'boost': {'allogroom': 0.07026408914940072,
   'dominancegroom': 0.04895076820557969},
  'penalize': {'sniff': 0.032264889275311445},
  'prefer': [('allogroom', 'sniff', 0.03540325922834654),
   ('dominancegroom', 'sniff', 0.03191512180892245)]},
 'CautiousGiraffe': {'boost': {'chase': 0.046873419525772175,
   'sniffbody': 0.04956469609402715},
  'penalize': {'sniffgenital': 0.026045976124565565},
  'prefer': [('chase', 'sniffgenital', 0.04373521192229624),
   ('sniffbody', 'reciprocalsniff', 0.042224058799378184)]},
 'ElegantMink': {'boost': {'allogroom': 0.051344438992756466,
   'ejaculate': 0.04419288246136156},
  'penalize': {'sniff': 0.028419975022486547},
  'prefer': [('allogroom', 'sniff', 0.04197073992987219),
   ('ejaculate', 'intromit', 0.03535669428927513)]},
 'NiftyGoldfinch': {'boost': {'tussle': 0.051902678013579284,
   'sniffgenital': 0.03182946515560948,
   'biteobject': 0.04947620077069683},
  'penalize': {'rear': 0.021486118867211898,
   'selfgroom': 0.012126113327424002},
  'prefer': [('tussle', 'defend', 0.033923677384710575),
   ('sniffgenital', 'sniff', 0.0441007770763866),
   ('biteobject', 'sniff', 0.035835949429890365)]},
 'AdaptableSnail': {'boost': {'chaseattack': 0.04, 'chase': 0.035},
  'penalize': {'avoid': 0.025},
  'prefer': [('chaseattack', 'chase', 0.035), ('chase', 'avoid', 0.035)]}}

TRAIN_LAB_ACTIONS = {
    "AdaptableSnail": [
        "approach", "attack", "avoid", "chase", "chaseattack", "rear", "submit"
    ],
    "BoisterousParrot": [
        "shepherd"
    ],
    "CRIM13": [
        "approach", "attack", "disengage", "mount", "rear",
        "selfgroom", "sniff"
    ],
    "CalMS21_supplemental": [
        "approach", "attemptmount", "attack", "dominancemount", "intromit",
        "mount", "sniff", "sniffbody", "sniffgenital", "sniffface"
    ],
    "CalMS21_task1": [
        "approach", "attack", "genitalgroom", "intromit", "mount",
        "sniff", "sniffbody", "sniffgenital", "sniffface"
    ],
    "CalMS21_task2": [
        "attack", "mount", "sniff"
    ],
    "CautiousGiraffe": [
        "chase", "escape", "reciprocalsniff", "sniff", "sniffbody", "sniffgenital"
    ],
    "DeliriousFly": [
        "attack", "dominance", "sniff"
    ],
    "ElegantMink": [
        "allogroom", "attack", "attemptmount", "ejaculate",
        "intromit", "mount", "sniff"
    ],
    "GroovyShrew": [
        "attemptmount", "climb", "defend", "dig", "escape", "rear",
        "rest", "run", "selfgroom", "sniff", "sniffgenital", "approach"
    ],
    "InvincibleJellyfish": [
        "allogroom", "attack", "dig", "dominancegroom", "escape",
        "selfgroom", "sniff", "sniffgenital"
    ],
    "JovialSwallow": [
        "attack", "chase", "sniff"
    ],
    "LyricalHare": [
        "approach", "attack", "defend", "escape", "freeze", "rear", "sniff"
    ],
    "NiftyGoldfinch": [
        "approach", "attack", "biteobject", "chase", "climb", "defend",
        "dig", "escape", "exploreobject", "flinch", "follow", "rear",
        "run", "selfgroom", "sniff", "sniffgenital", "sniffface", "tussle"
    ],
    "PleasantMeerkat": [
        "attack", "chase", "escape", "follow"
    ],
    "ReflectiveManatee": [
        "attack", "sniff"
    ],
    "SparklingTapir": [
        "attack", "defend", "escape", "mount"
    ],
    "TranquilPanther": [
        "intromit", "mount", "rear", "selfgroom", "sniff", "sniffgenital"
    ],
    "UppityFerret": [
        "huddle", "reciprocalsniff", "sniffgenital"
    ]
}

SELF_ACTIONS = ['rear', 'selfgroom', 'genitalgroom', 'rest', 'climb', 'dig', 'run', 'freeze', 'biteobject', 'exploreobject', 'huddle']
PAIR_ACTIONS = ['approach', 'attack', 'avoid', 'chase', 'chaseattack', 'submit', 'shepherd', 'disengage', 'mount', 'sniff', 'sniffgenital', 'dominancemount', 'sniffbody', 'sniffface', 'attemptmount', 'intromit', 'escape', 'reciprocalsniff', 'dominance', 'allogroom', 'ejaculate', 'defend', 'dominancegroom', 'flinch', 'follow', 'tussle']

# =============================================================================
# 2. Helper Functions
# =============================================================================

def smooth_probs_inplace(probs: pd.DataFrame, actions: List[str], win: int = 5):
    if win is None or win <= 1 or probs.empty: 
        return
    act_cols = [f"action_{a}" for a in actions]
    # Ensure sorted
    probs.sort_values(["video_id","agent_id","target_id","frame"], inplace=True)
    pad = win // 2
    kernel = np.ones(win, dtype=np.float32) / win
    gobj = probs.groupby(["video_id","agent_id","target_id"], sort=False)
    for _, idx in gobj.groups.items():
        idx = np.asarray(idx)
        arr = probs.loc[idx, act_cols].to_numpy(np.float32, copy=False)
        for j in range(arr.shape[1]):
            v = arr[:, j]
            if v.size:
                vp = np.pad(v, (pad, pad), mode="edge")
                arr[:, j] = np.convolve(vp, kernel, mode="valid").astype(np.float32, copy=False)
        probs.loc[idx, act_cols] = arr

def mask_probs_numpy_rle(probs: pd.DataFrame, ACTIONS: List[str], active_map: Dict[int, Set[str]],
                         copy=True) -> pd.DataFrame:
    df = probs.copy() if copy else probs
    if not len(df): return df
    action_cols = [f"action_{a}" for a in ACTIONS]
    act_block = df[action_cols].to_numpy(copy=False)
    N, A = act_block.shape
    vid = df["video_id"].to_numpy(np.int64, copy=False)
    ag  = df["agent_id"].to_numpy(np.int64, copy=False)
    tg  = df["target_id"].to_numpy(np.int64, copy=False)

    act_pos = {a: i for i, a in enumerate(ACTIONS)}
    allow: Dict[int, Dict[Tuple[int,int], np.ndarray]] = {}
    for v, triples in active_map.items():
        v = int(v)
        d = allow.setdefault(v, {})
        for s in triples:
            sag, stg, sa = s.split(",")
            key = (int(sag), int(stg))
            arr = d.get(key)
            if arr is None:
                arr = np.zeros(A, dtype=bool)
                d[key] = arr
            i = act_pos.get(sa)
            if i is not None:
                arr[i] = True

    if N == 1:
        starts = np.array([0], dtype=np.int64); ends = np.array([1], dtype=np.int64)
    else:
        change = (vid[1:] != vid[:-1]) | (ag[1:] != ag[:-1]) | (tg[1:] != tg[:-1])
        boundaries = np.flatnonzero(change) + 1
        starts = np.concatenate(([0], boundaries))
        ends   = np.concatenate((boundaries, [N]))

    for s, e in zip(starts, ends):
        v, a_, t_ = int(vid[s]), int(ag[s]), int(tg[s])
        d = allow.get(v)
        if d is None:
            act_block[s:e, :] = 0.0; continue
        mask = d.get((a_, t_))
        if mask is None:
            act_block[s:e, :] = 0.0; continue
        if not mask.all():
            disallowed = ~mask
            act_block[s:e, disallowed] = 0.0

    df[action_cols] = act_block
    return df

def probs_to_nonoverlapping_intervals(
    prob_df: pd.DataFrame,
    actions: List[str],
    min_len: int = 3,
    max_gap: int = 2,
    lab: str | None = None,
    tie_config: Dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Convert frame-level probs to non-overlapping intervals for a *single lab*.
    """
    out: list[dict] = []
    act_cols = [f"action_{a}" for a in actions]

    # Per-lab thresholds
    per_action_thresh = TT_PER_LAB_NN.get(lab, {})
    
    # Default threshold 0.75 if not specified
    thr = np.array(
        [per_action_thresh.get(a, 0.75) for a in actions],
        dtype=np.float32,
    )

    # Optional per-lab tie rules
    lab_tie_cfg = None
    if tie_config is not None and lab is not None and lab in tie_config:
        lab_tie_cfg = tie_config[lab]
        action_to_idx = {a: i for i, a in enumerate(actions)}
    else:
        action_to_idx = {}

    for (vid, ag, tg), grp in prob_df.groupby(["video_id", "agent_id", "target_id"], sort=False):
        g = grp.sort_values("frame")
        frames = g["frame"].to_numpy()
        P = g[act_cols].to_numpy(np.float32)  # [T, num_actions]

        pass_mask = (P >= thr[None, :])

        # ---- Tie manipulation (per lab) ----
        P_adj = P.copy()
        if lab_tie_cfg is not None:
            multi_mask = (pass_mask.sum(axis=1) > 1)

            if multi_mask.any():
                # 1) boosts
                boost_cfg = lab_tie_cfg.get("boost", {})
                for act, delta in boost_cfg.items():
                    idx = action_to_idx.get(act, None)
                    if idx is not None:
                        P_adj[multi_mask, idx] += float(delta)

                # 2) penalties
                penalize_cfg = lab_tie_cfg.get("penalize", {})
                for act, delta in penalize_cfg.items():
                    idx = action_to_idx.get(act, None)
                    if idx is not None:
                        P_adj[multi_mask, idx] -= float(delta)

                # 3) explicit preferences
                prefer_cfg = lab_tie_cfg.get("prefer", [])
                for winner_act, loser_act, margin in prefer_cfg:
                    wi = action_to_idx.get(winner_act, None)
                    li = action_to_idx.get(loser_act, None)
                    if wi is None or li is None:
                        continue
                    fm = multi_mask & pass_mask[:, wi] & pass_mask[:, li]
                    if fm.any():
                        P_adj[fm, wi] += float(margin)

                np.clip(P_adj, 0.0, 1.0, out=P_adj)

        # --- Best-label decoding ---
        P_masked = np.where(pass_mask, P_adj, -np.inf)
        best_idx = np.argmax(P_masked, axis=1)
        best_val = P_masked[np.arange(len(P_masked)), best_idx]
        label = np.where(np.isfinite(best_val), best_idx, -1)

        # --- Fill short gaps ---
        if max_gap > 0:
            i = 0
            while i < len(label):
                if label[i] >= 0:
                    j = i
                    while j + 1 < len(label) and label[j + 1] == label[i]:
                        j += 1
                    k = j + 1
                    while k < len(label) and label[k] == -1:
                        k += 1
                    if k < len(label) and label[k] == label[i] and (k - j - 1) <= max_gap:
                        label[j + 1:k] = label[i]
                        j = k
                    i = j + 1
                else:
                    i += 1

        # --- Convert label sequence to intervals ---
        def flush(s, e, idx):
            if s is None:
                return
            if frames[e] - frames[s] + 1 >= min_len:
                out.append(
                    {
                        "video_id": int(vid),
                        "agent_id": int(ag),
                        "target_id": int(tg),
                        "action": actions[idx],
                        "start_frame": int(frames[s]),
                        "stop_frame": int(frames[e] + 1),
                    }
                )

        s = None
        cur = -1
        for i_t, idx in enumerate(label):
            if idx != cur:
                if cur >= 0:
                    flush(s, i_t - 1, cur)
                s = i_t if idx >= 0 else None
                cur = idx
        if cur >= 0:
            flush(s, len(label) - 1, cur)

    return pd.DataFrame(out)
