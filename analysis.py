import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


DATA_DIR = "data"
FIG_DIR = "figures"
OUT_DIR = "outputs"


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    elect = pd.read_csv(os.path.join(DATA_DIR, "2020_US_County_Level_Presidential_Results.csv"))
    county = pd.read_csv(os.path.join(DATA_DIR, "County_demographics.csv"))
    state = pd.read_csv(os.path.join(DATA_DIR, "StateNameData.csv"))
    return elect, county, state


def merge_core(elect: pd.DataFrame, county: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    st = state[["State", "Code"]].rename(columns={"State": "state_name", "Code": "state_code"})
    elect2 = elect.merge(st, on="state_name", how="left")

    merged = elect2.merge(
        county[["County", "State", "Population.Population per Square Mile"]],
        left_on=["county_name", "state_code"],
        right_on=["County", "State"],
        how="inner"
    ).copy()

    merged["Log_Pop_SqMi"] = np.log10(merged["Population.Population per Square Mile"].replace({0: np.nan}))
    return merged


def make_density_margin_table(merged: pd.DataFrame) -> pd.DataFrame:
    tbl = (
        merged[["County", "state_name", "per_point_diff", "Log_Pop_SqMi"]]
        .dropna()
        .rename(columns={"state_name": "State"})
        .sort_values("per_point_diff", ascending=True)
        .reset_index(drop=True)
    )
    tbl.to_csv(os.path.join(OUT_DIR, "density_margin_table.csv"), index=False)
    return tbl


def simple_regression_and_plot(tbl: pd.DataFrame):
    x = tbl[["Log_Pop_SqMi"]].values
    y = tbl["per_point_diff"].values

    lm = LinearRegression().fit(x, y)
    m = float(lm.coef_[0])
    b = float(lm.intercept_)
    r = float(np.corrcoef(tbl["Log_Pop_SqMi"], tbl["per_point_diff"])[0, 1])

    # Scatter + fit line
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        tbl["Log_Pop_SqMi"], tbl["per_point_diff"],
        c=tbl["per_point_diff"], cmap="coolwarm", vmin=-1, vmax=1, s=6
    )
    plt.xlabel("Log Population per Square Mile")
    plt.ylabel("per_point_diff")
    plt.ylim(-1, 1)
    plt.title("Population Density vs. Election Margin")
    plt.colorbar(sc, label="per_point_diff")

    xs = np.linspace(tbl["Log_Pop_SqMi"].min(), tbl["Log_Pop_SqMi"].max(), 200)
    plt.plot(xs, m * xs + b)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "density_vs_margin.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    summary = {"correlation_r": r, "slope_m": m, "intercept_b": b, "figure": fig_path}
    pd.DataFrame([summary]).to_json(os.path.join(OUT_DIR, "simple_regression_summary.json"), orient="records", indent=2)
    print(f"[Simple Regression] r={r:.3f}, m={m:.3f}, b={b:.3f} -> {fig_path}")


def feature_correlations(merged: pd.DataFrame, county: pd.DataFrame):
    # numeric columns from county that are present in merged
    county_numeric = county.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in county_numeric if c in merged.columns]
    cols = sorted(set(cols + ["Log_Pop_SqMi"]))  # include derived feature

    out = {}
    for c in cols:
        sub = merged[["per_point_diff", c]].dropna()
        if len(sub) >= 2:
            out[c] = float(sub.corr().iloc[0, 1])

    s = pd.Series(out).sort_values(ascending=True)
    path = os.path.join(OUT_DIR, "feature_correlations.csv")
    s.to_csv(path, header=["corr_with_per_point_diff"])
    print(f"[Correlations] saved -> {path}")
    return s


def multivariate_fit_and_residuals(merged: pd.DataFrame, county: pd.DataFrame):
    # Build feature matrix from county numeric cols + Log_Pop_SqMi
    county_numeric = county.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in county_numeric if c in merged.columns]
    if "Log_Pop_SqMi" not in features:
        features.append("Log_Pop_SqMi")

    Xfull = merged[features].replace([np.inf, -np.inf], np.nan)
    mask = Xfull.notna().all(axis=1) & merged["per_point_diff"].notna()
    X = Xfull.loc[mask].values
    y = merged.loc[mask, "per_point_diff"].values

    lm = LinearRegression().fit(X, y)
    pred = lm.predict(X)
    resid = y - pred

    res_df = (
        pd.DataFrame({
            "County": merged.loc[mask, "County"].values,
            "State": merged.loc[mask, "state_name"].values,  # full state name
            "per_point_diff": y,
            "prediction": pred,
            "residual": resid
        })
        .sort_values("residual", ascending=True)
        .reset_index(drop=True)
    )

    out_csv = os.path.join(OUT_DIR, "residuals_sorted.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"[Multivariate] residuals saved -> {out_csv}")

    # Pred vs actual plot, colored by residual
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(res_df["per_point_diff"], res_df["prediction"],
                     c=res_df["residual"], cmap="bwr", vmin=-1, vmax=1, s=6)
    plt.xlabel("per_point_diff")
    plt.ylabel("prediction")
    plt.xlim(-1, 1)
    lims = [-1, 1]
    plt.plot(lims, lims)
    plt.colorbar(sc, label="residual")
    plt.title("Predicted vs Actual (colored by residual)")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "predicted_vs_actual.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[Multivariate] figure saved -> {fig_path}")


def main():
    ensure_dirs()
    elect, county, state = load_data()
    merged = merge_core(elect, county, state)

    tbl = make_density_margin_table(merged)           # sorted table export
    simple_regression_and_plot(tbl)                   # coolwarm + colorbar + line
    feature_correlations(merged, county)              # full correlation sweep
    multivariate_fit_and_residuals(merged, county)    # residuals csv + plot


if __name__ == "__main__":
    main()
