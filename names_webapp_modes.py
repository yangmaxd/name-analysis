# names_webapp_modes.py
# Streamlit app: two modes (Name search / Year search), gender-aware Top-20 lists,
# overlay plotting, normalization, and age stats.
#
# Run:
#   pip install streamlit plotly pandas numpy
#   streamlit run names_webapp_modes.py

from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Names Analyzer — Modes", layout="wide")
st.title("Names Analyzer — Web (Name search & Year search modes)")

with st.expander("How to use"):
    st.markdown(
        """
        **Data**: Upload a CSV with at least `name, year, count`. If you also include `sex`/`gender` (M/F), the **Year search** mode can show **Top‑20 boys** and **Top‑20 girls** for a chosen year.
        
        **Modes:**
        - **Name search** — type or select names, overlay their trends, optionally normalize, and see age stats.
        - **Year search** — pick a year; get Top‑20 boys and Top‑20 girls (if gender exists), select which to plot.
        """
    )

# ---------------- Helpers ----------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common column aliases to: name, year, count, sex (optional)."""
    lower_cols = {c.lower(): c for c in df.columns}
    rename = {}
    # required
    if "name" in lower_cols: rename[lower_cols["name"]] = "name"
    if "year" in lower_cols: rename[lower_cols["year"]] = "year"
    if "count" in lower_cols: rename[lower_cols["count"]] = "count"
    # alternates
    for alt, std in [
        ("given","name"), ("firstname","name"), ("hangul","name"), ("name_kr","name"),
        ("birth_year","year"),
        ("freq","count"), ("frequency","count"),
        ("sex","sex"), ("gender","sex")
    ]:
        if std not in rename and alt in lower_cols:
            rename[lower_cols[alt]] = std
    out = df.rename(columns=rename).copy()

    # Normalize sex values if present
    if "sex" in out.columns:
        out["sex"] = out["sex"].astype(str).str.strip().str.upper().map({
            "M":"M","MALE":"M","BOY":"M","B":"M",
            "F":"F","FEMALE":"F","GIRL":"F","G":"F"
        }).fillna(np.nan)
    return out

def weighted_age_stats(years: np.ndarray, counts: np.ndarray, ref_year: float):
    counts = np.asarray(counts, dtype=float)
    years = np.asarray(years, dtype=float)
    mask = np.isfinite(years) & np.isfinite(counts) & (counts >= 0)
    years, counts = years[mask], counts[mask]
    if len(years) == 0 or counts.sum() <= 0:
        return None
    order = np.argsort(years)
    years = years[order]; counts = counts[order]
    total = counts.sum()
    probs = counts / total
    mean_by = float(np.sum(years * probs))
    var_by = float(np.sum((years - mean_by) ** 2 * probs))
    std_by = var_by ** 0.5
    cdf = np.cumsum(probs)
    p025_idx = np.searchsorted(cdf, 0.025, side="left")
    p975_idx = np.searchsorted(cdf, 0.975, side="left")
    p025_by = float(years[min(p025_idx, len(years)-1)])
    p975_by = float(years[min(p975_idx, len(years)-1)])
    mean_age = float(ref_year - mean_by)
    age_low_95 = float(ref_year - p975_by)
    age_high_95 = float(ref_year - p025_by)
    r1 = lambda x: float(np.round(x, 1))
    return {"mean_age": r1(mean_age), "std_age": r1(std_by),
            "age_95_low": r1(age_low_95), "age_95_high": r1(age_high_95),
            "mean_birth_year": r1(mean_by)}

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload a CSV (expects: name, year, count; optional: sex/gender)", type=["csv"])
if not uploaded:
    st.info("Upload a CSV to begin."); st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}"); st.stop()

df = standardize_columns(raw).dropna(subset=["name"])

# Validate required columns
if not set(["name","year","count"]).issubset(set([c.lower() for c in df.columns])) and not set(["name","year","count"]).issubset(set(df.columns)):
    st.error("CSV must include columns that map to: name, year, count."); st.stop()

# types & basic cleaning
df["name"] = df["name"].astype(str).str.strip()
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["count"] = pd.to_numeric(df["count"], errors="coerce")
df = df.dropna(subset=["year","count"])
df = df[(df["year"] > 0) & (df["count"] >= 0)]

has_sex = "sex" in df.columns and df["sex"].notna().any()

st.caption(f"Rows: **{len(df):,}** • Names: **{df['name'].nunique():,}** • Years: {int(df['year'].min())}–{int(df['year'].max())} • Sex column: {'Yes' if has_sex else 'No'}")

# ---------------- Global controls ----------------
normalize = st.sidebar.checkbox("Normalize each curve (0–1)", value=False)
ref_year = st.sidebar.number_input("Age reference year", min_value=1800, max_value=2200, value=2025, step=1)

# ---------------- Mode selection ----------------
mode = st.radio("Mode", ["Name search", "Year search"], horizontal=True)

selection: list[str] = []  # names to plot

if mode == "Name search":
    st.subheader("Name search mode")
    left, right = st.columns([1,2])
    with left:
        all_names = sorted(df["name"].unique().tolist())
        # Select or type names (comma-separated)
        picked = st.multiselect("Select names", options=all_names, default=all_names[:1])
        type_in = st.text_input("...and/or type names (comma-separated)", value="", placeholder="e.g., Mary, John, Sophia")
        typed = [x.strip() for x in type_in.split(",") if x.strip()]
        # Validate typed names: keep only those that exist (case-insensitive)
        name_set_lower = {n.lower(): n for n in all_names}
        valid_typed = []
        for nm in typed:
            key = nm.lower()
            if key in name_set_lower:
                valid_typed.append(name_set_lower[key])
        if typed and not valid_typed:
            st.warning("Typed names not found in this dataset (case-insensitive).")
        selection = sorted(set(picked) | set(valid_typed))

    with right:
        if selection:
            sub = df[df["name"].isin(selection)].copy().sort_values(["name","year"])
            if normalize:
                sub["plot_y"] = sub.groupby("name")["count"].transform(lambda s: s/s.max() if s.max()>0 else s)
                ycol, ylab = "plot_y", "Count (normalized)"
            else:
                ycol, ylab = "count", "Count"
            fig = px.line(sub, x="year", y=ycol, color="name", markers=True,
                          title="Count vs Year", labels={"year":"Year", ycol: ylab})
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select or type at least one name to plot.")

elif mode == "Year search":
    st.subheader("Year search mode")
    year = st.number_input("Choose a year", min_value=int(df["year"].min()), max_value=int(df["year"].max()), value=int(df["year"].median()), step=1)
    yr_df = df[df["year"] == year]

    if has_sex:
        boys = (yr_df[yr_df["sex"] == "M"].groupby("name", as_index=False)["count"].sum()
                .sort_values("count", ascending=False).head(20))
        girls = (yr_df[yr_df["sex"] == "F"].groupby("name", as_index=False)["count"].sum()
                 .sort_values("count", ascending=False).head(20))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 20 boys**")
            boys_sel = st.multiselect("Select boys to plot", options=boys["name"].tolist(), default=boys["name"].tolist()[:5], key="boys_sel")
        with col2:
            st.markdown("**Top 20 girls**")
            girls_sel = st.multiselect("Select girls to plot", options=girls["name"].tolist(), default=girls["name"].tolist()[:5], key="girls_sel")
        selection = sorted(set(boys_sel) | set(girls_sel))
    else:
        st.info("No sex/gender column detected. Showing overall Top 40 for this year.")
        top40 = (yr_df.groupby("name", as_index=False)["count"].sum()
                 .sort_values("count", ascending=False).head(40))
        selection = st.multiselect("Select names to plot", options=top40["name"].tolist(), default=top40["name"].tolist()[:10])

    # Plot selections (across all years)
    if selection:
        sub = df[df["name"].isin(selection)].copy().sort_values(["name","year"])
        if normalize:
            sub["plot_y"] = sub.groupby("name")["count"].transform(lambda s: s/s.max() if s.max()>0 else s)
            ycol, ylab = "plot_y", "Count (normalized)"
        else:
            ycol, ylab = "count", "Count"
        fig = px.line(sub, x="year", y=ycol, color="name", markers=True,
                      title="Count vs Year", labels={"year":"Year", ycol: ylab})
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pick at least one name to plot.")

# ---------------- Stats ----------------
st.subheader("Age statistics (weighted by counts)")
rows = []
for nm in selection or []:
    s = df[df["name"] == nm].sort_values("year")
    stv = weighted_age_stats(s["year"].to_numpy(), s["count"].to_numpy(), ref_year=ref_year)
    if stv:
        rows.append({"name": nm, **stv})
if rows:
    st.dataframe(pd.DataFrame(rows))
else:
    st.caption("Select names to compute age stats.")

