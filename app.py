"""
Motor insurance premium calculator.

Uses the tariff (base premium + relativities) exported by
notebooks/motor_pricing.ipynb. Run with:  streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from prep import add_bands  # same banding as the model

tariff = json.loads((ROOT / "models" / "tariff.json").read_text())
FACTORS = tariff["factors"]

st.set_page_config(page_title="Motor Premium Calculator", page_icon="🚗", layout="wide")
st.title("🚗 Motor Insurance Premium Calculator")
st.caption(
    "Pure premium from Poisson (frequency) × Gamma (severity) GLMs fitted to 678k French "
    "motor policies (freMTPL2). Educational model only, not a real insurance quote."
)

# ---------------------------------------------------------------- inputs
left, right = st.columns([1, 1.4])
with left:
    st.subheader("Driver and vehicle")
    driv_age = st.slider("Driver age", 18, 90, 35)
    bonus_malus = st.slider(
        "Bonus-malus score", 50, 150, 50,
        help="French no-claims system: starts at 100, falls 5% each claim-free year to a minimum "
             "of 50, rises 25% per at-fault claim. Lower = better history.",
    )
    veh_age = st.slider("Vehicle age (years)", 0, 30, 4)
    veh_power = st.slider("Vehicle power class", 4, 15, 6)
    veh_gas = st.radio("Fuel", ["Regular", "Diesel"], horizontal=True)
    area = st.select_slider(
        "Area (population density)", options=list("ABCDEF"), value="C",
        help="A = rural ... F = dense urban",
    )
    region = st.selectbox(
        "Region", sorted(FACTORS["Region"], key=lambda r: (r != tariff["base_levels"]["Region"], r))
    )
    st.subheader("Commercial loading")
    loading = st.slider(
        "Expenses, commission and profit (% of premium)", 0, 50, 30,
        help="Office premium = pure premium / (1 - loading).",
    ) / 100

# ---------------------------------------------------------------- calculation
profile = pd.DataFrame([{
    "DrivAge": driv_age, "VehAge": veh_age, "BonusMalus": bonus_malus, "VehPower": veh_power,
    "VehGas": veh_gas, "Area": area, "Region": region,
}])
banded = add_bands(profile).iloc[0]

rows, freq, sev = [], tariff["base_frequency"], tariff["base_severity"]
for f in FACTORS:
    level = banded[f]
    rel = FACTORS[f][level]
    freq *= rel["freq"]
    sev *= rel["sev"]
    rows.append({"Factor": f, "Your band": level,
                 "Frequency ×": rel["freq"], "Severity ×": rel["sev"], "Combined ×": rel["combined"]})

pure_premium = freq * sev * tariff["large_loss_load"]
office_premium = pure_premium / (1 - loading)

with right:
    st.subheader("Your quote")
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected claims / year", f"{freq:.3f}", help="Poisson GLM")
    c2.metric("Expected cost / claim", f"€{sev * tariff['large_loss_load']:,.0f}",
              help="Gamma GLM, including the large-loss loading")
    c3.metric("Pure premium", f"€{pure_premium:,.0f}")
    st.metric("Office premium (incl. loading)", f"€{office_premium:,.0f} per year",
              delta=f"{pure_premium / tariff['base_premium'] - 1:+.0%} vs base risk",
              delta_color="inverse")

    st.markdown("**How the premium is built**")
    st.markdown(
        f"Base pure premium **€{tariff['base_premium']:,.2f}** "
        f"(a 41–50 year-old, bonus-malus 50, 3–5 year-old power-6 petrol car, area C, region R24) "
        f"× the multiplier for each of your bands:"
    )
    st.dataframe(
        pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ["Frequency ×", "Severity ×", "Combined ×"]})
        .background_gradient(subset=["Combined ×"], cmap="RdYlGn_r", vmin=0.5, vmax=2),
        hide_index=True, use_container_width=True,
    )

    with st.expander("Model performance (test set)"):
        st.dataframe(pd.DataFrame(tariff["metrics"]).T, use_container_width=True)
        st.caption("Gini measures how well the model ranks risk (0 = random). "
                   "The GLM is used for pricing because each factor is a transparent multiplier.")
