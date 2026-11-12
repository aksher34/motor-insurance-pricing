"""
Data preparation for the freMTPL2 motor pricing project.

Shared by the analysis notebook and the Streamlit app so that both use
exactly the same rating-factor definitions (bands).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ---------------------------------------------------------------------------
# Rating-factor bands
# ---------------------------------------------------------------------------
# Continuous variables are grouped into bands. This is standard in pricing:
# it lets the GLM capture non-linear effects (e.g. young AND old drivers being
# riskier) and produces a simple table of multipliers per band.

DRIV_AGE_BINS = [17, 20, 25, 30, 40, 50, 60, 70, 120]
DRIV_AGE_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"]

VEH_AGE_BINS = [-1, 0, 2, 5, 10, 15, 200]
VEH_AGE_LABELS = ["0", "1-2", "3-5", "6-10", "11-15", "16+"]

BONUS_MALUS_BINS = [0, 50, 60, 80, 100, 400]
BONUS_MALUS_LABELS = ["50", "51-60", "61-80", "81-100", "101+"]

VEH_POWER_BINS = [0, 4, 5, 6, 7, 8, 100]
VEH_POWER_LABELS = ["4", "5", "6", "7", "8", "9+"]

# Base (reference) level for each factor: the most common / lowest-risk
# group, so every other level's multiplier is read relative to it.
BASE_LEVELS = {
    "DrivAgeBand": "41-50",
    "VehAgeBand": "3-5",
    "BonusMalusBand": "50",
    "VehPowerBand": "6",
    "VehGas": "Regular",
    "Area": "C",
    "Region": "R24",
}

RATING_FACTORS = list(BASE_LEVELS.keys())

# Natural display order of each banded factor (for charts and tables)
BAND_ORDER = {
    "DrivAgeBand": DRIV_AGE_LABELS,
    "VehAgeBand": VEH_AGE_LABELS,
    "BonusMalusBand": BONUS_MALUS_LABELS,
    "VehPowerBand": VEH_POWER_LABELS,
}


def level_order(factor: str, levels) -> list:
    """Sort a factor's levels in their natural order (bands) or alphabetically."""
    levels = list(levels)
    if factor in BAND_ORDER:
        return [lvl for lvl in BAND_ORDER[factor] if lvl in levels]
    return sorted(levels)

# Individual claims above this amount are capped; the excess is spread across
# all policies as a "large-loss loading" (see notebook section 5).
LARGE_CLAIM_CAP = 50_000


def add_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add banded rating factors to a policy-level dataframe."""
    out = df.copy()
    out["DrivAgeBand"] = pd.cut(out["DrivAge"], DRIV_AGE_BINS, labels=DRIV_AGE_LABELS)
    out["VehAgeBand"] = pd.cut(out["VehAge"], VEH_AGE_BINS, labels=VEH_AGE_LABELS)
    out["BonusMalusBand"] = pd.cut(out["BonusMalus"], BONUS_MALUS_BINS, labels=BONUS_MALUS_LABELS)
    out["VehPowerBand"] = pd.cut(out["VehPower"], VEH_POWER_BINS, labels=VEH_POWER_LABELS)
    for col in RATING_FACTORS:
        out[col] = out[col].astype(str)
    return out


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the raw data.

    Returns
    -------
    policies : one row per policy, with claim counts, capped claim totals and bands
    claims   : one row per individual claim (raw amounts), for severity analysis
    """
    freq = pd.read_csv(DATA_DIR / "freMTPL2freq.csv")
    sev = pd.read_csv(DATA_DIR / "freMTPL2sev.csv")

    freq["IDpol"] = freq["IDpol"].astype(int)
    for col in ["Area", "VehGas", "Region", "VehBrand"]:
        freq[col] = freq[col].astype(str).str.strip("'")

    # --- Cleaning -----------------------------------------------------------
    # 1,224 policies have exposure > 1 year, which isn't possible for an
    # annual contract (data error) -> cap at 1.
    freq["Exposure"] = freq["Exposure"].clip(upper=1.0)

    # The policy file records 36,102 claims, but only 26,639 have an amount in
    # the claims file (9,116 policies show claims with no payment recorded -
    # most likely claims closed at nil cost). For a PURE PREMIUM model we only
    # care about claims that cost money, so claim counts are rebuilt from the
    # claims file. This keeps the frequency and severity models consistent.
    # (Judgement call - discussed in the README.)
    claims_per_pol = sev.groupby("IDpol").agg(
        ClaimNb=("ClaimAmount", "size"),
        ClaimAmountRaw=("ClaimAmount", "sum"),
    )
    sev["ClaimAmountCapped"] = sev["ClaimAmount"].clip(upper=LARGE_CLAIM_CAP)
    claims_per_pol["ClaimAmountCapped"] = sev.groupby("IDpol")["ClaimAmountCapped"].sum()

    policies = freq.drop(columns="ClaimNb").merge(
        claims_per_pol, left_on="IDpol", right_index=True, how="left"
    )
    policies[["ClaimNb", "ClaimAmountRaw", "ClaimAmountCapped"]] = (
        policies[["ClaimNb", "ClaimAmountRaw", "ClaimAmountCapped"]].fillna(0)
    )
    policies["ClaimNb"] = policies["ClaimNb"].astype(int)

    # A tiny number of policies show 5+ claims in a year, which is very likely
    # a data issue. Cap the count at 4 (common practice for this dataset).
    policies["ClaimNb"] = policies["ClaimNb"].clip(upper=4)

    policies = add_bands(policies)

    # Keep only claims that belong to a policy in the policy file.
    claims = sev[sev["IDpol"].isin(policies["IDpol"])].copy()

    return policies, claims


def gini(actual: np.ndarray, predicted: np.ndarray, exposure: np.ndarray) -> float:
    """
    Exposure-weighted Gini coefficient (how well the model ranks risks).

    Sort policies from lowest to highest predicted risk, then plot the
    cumulative share of exposure (x) against the cumulative share of actual
    claims (y). A model with no skill gives the diagonal (Gini = 0); the
    further the curve bows below the diagonal, the better the ranking.
    """
    order = np.argsort(predicted)
    cum_exposure = np.cumsum(exposure[order]) / exposure.sum()
    cum_claims = np.cumsum(actual[order]) / actual.sum()
    cum_exposure = np.concatenate([[0], cum_exposure])
    cum_claims = np.concatenate([[0], cum_claims])
    area = np.trapezoid(cum_claims, cum_exposure)
    return 1 - 2 * area
