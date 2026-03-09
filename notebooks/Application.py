import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px  # ✅ hover/zoom/tooltips

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="Chômage France", layout="wide")

DEFAULT_CSV_PATH = "../data/processed/02_dataset_silver.csv"
TARGET = "taux_chomage_total_insee"
DATE_COL = "date"
DROP_COLS_ALWAYS = ["taux_chomage_ocde"]  # demandé : on l'enlève


# --------------------------
# HELPERS
# --------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def prepare_xy(df: pd.DataFrame):
    drop_cols = [c for c in [TARGET, DATE_COL] + DROP_COLS_ALWAYS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[TARGET] if TARGET in df.columns else None
    return X, y


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage robuste des colonnes numériques pour éviter inf / valeurs énormes."""
    d = df.copy()
    num_cols = d.select_dtypes(include=[np.number]).columns
    d[num_cols] = d[num_cols].apply(pd.to_numeric, errors="coerce")
    d[num_cols] = d[num_cols].replace([np.inf, -np.inf], np.nan)
    max_float = np.finfo(np.float64).max
    d[num_cols] = d[num_cols].mask(d[num_cols].abs() > max_float, np.nan)
    return d


def find_age_cols(df: pd.DataFrame):
    """
    Détecte les colonnes chômage par âge, y compris 55+ / 50+ etc.
    Ex: taux_chomage_15_24_insee, taux_chomage_25_49_insee, taux_chomage_50_64_insee, taux_chomage_55_64_insee...
    """
    cols = []
    for c in df.columns:
        name = c.lower()
        if "taux_chomage" in name and "insee" in name:
            if c == TARGET or c in DROP_COLS_ALWAYS:
                continue
            if any(
                k in name
                for k in [
                    "15_24",
                    "16_24",
                    "25_49",
                    "25_54",
                    "50_64",
                    "55_64",
                    "plus",
                    "moins",
                ]
            ):
                cols.append(c)
    return cols


def pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def line_plot_interactive(dff: pd.DataFrame, y_cols, title, y_label="Valeur"):
    """
    Line chart interactif Plotly avec hover.
    y_cols peut être un str ou une liste de colonnes.
    """
    fig = px.line(
        dff,
        x=DATE_COL,
        y=y_cols,
        title=title,
        labels={"value": y_label, "variable": "Série"},
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("Chômage France 🇫🇷")
st.sidebar.caption("Évolution du chômage en France (2000-2025)")
csv_path = "../data/processed/02_dataset_silver.csv"
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 Dashboard", "📊 EDA", "🧠 Feature Importance", "🔮 Prédiction"],
)

# --------------------------
# LOAD
# --------------------------
try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"Impossible de charger le fichier: {e}")
    st.stop()

df = clean_numeric(df)
st.sidebar.success(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")

if TARGET not in df.columns or DATE_COL not in df.columns:
    st.error("Il manque la colonne cible (TARGET) ou la colonne date (DATE_COL).")
    st.stop()


# =========================================================
# PAGE ACCUEIL
# =========================================================
if page == "🏠 Accueil":
    st.title("Analyse du chômage en France")
    st.write(
        "Cette application permet de visualiser les indicateurs clés du chômage en France "
        "et de comparer l’évolution selon le sexe et l’âge, puis d’identifier les variables "
        "les plus importantes et tester une prédiction simple."
    )

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown(f"**Cible (TARGET)** : `{TARGET}`")
# =========================================================
# PAGE DASHBOARD
# =========================================================
elif page == "📊 Dashboard":
    st.title("📊 Dashboard — Indicateurs clés (interactif)")

    # ------- filtres AU-DESSUS -------
    d0 = df.dropna(subset=[DATE_COL]).copy()
    min_d = d0[DATE_COL].min().date()
    max_d = d0[DATE_COL].max().date()

    st.subheader("Filtres")
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "Date début",
            value=min_d,
            min_value=min_d,
            max_value=max_d,
            key="dash_start",
        )
    with c2:
        end_date = st.date_input(
            "Date fin", value=max_d, min_value=min_d, max_value=max_d, key="dash_end"
        )

    dff = df.dropna(subset=[DATE_COL]).copy()
    dff = dff[
        (dff[DATE_COL] >= pd.to_datetime(start_date))
        & (dff[DATE_COL] <= pd.to_datetime(end_date))
    ].sort_values(DATE_COL)

    if dff.empty:
        st.warning("Aucune donnée après filtres.")
        st.stop()

    st.divider()

    # =====================================================
    # INDICATEUR 1 : EVOLUTION GLOBALE
    # =====================================================
    st.subheader("1) Évolution du taux de chômage (global)")
    line_plot_interactive(
        dff, TARGET, "Évolution du taux de chômage total (INSEE)", y_label="Taux (%)"
    )
    st.write(
        "Cet indicateur montre l’évolution du chômage global sur la période sélectionnée, "
        "ce qui permet d’identifier rapidement les phases de hausse/baisse et la tendance générale."
    )

    st.divider()

    # =====================================================
    # INDICATEUR 2 : CHOMAGE PAR SEXE
    # =====================================================
    st.subheader("2) Taux de chômage selon le sexe")
    col_h = "taux_chomage_homme_insee"
    col_f = "taux_chomage_femme_insee"

    if col_h in df.columns and col_f in df.columns:
        line_plot_interactive(
            dff,
            [col_h, col_f],
            "Comparaison du chômage entre hommes et femmes",
            y_label="Taux (%)",
        )
        st.write(
            "Cet indicateur compare le chômage chez les hommes et chez les femmes afin d’observer "
            "d’éventuels écarts sur le marché du travail et leur évolution dans le temps."
        )
    else:
        st.info(
            "Colonnes sexe non trouvées (attendues: taux_chomage_homme_insee et taux_chomage_femme_insee)."
        )

    st.divider()

    # =====================================================
    # INDICATEUR 3 : JEUNES vs ADULTES
    # =====================================================
    st.subheader("3) Chômage — Tranches d'ages")

    col_jeunes = pick_first_existing(
        df, ["taux_chomage_15_24_insee", "taux_chomage_moins25_insee"]
    )
    col_adultes = pick_first_existing(
        df, ["taux_chomage_25_49_insee", "taux_chomage_25_54_insee"]
    )
    col_vieux = pick_first_existing(
        df, ["taux_chomage_50_plus_insee", "taux_chomage_50_plus_insee"]
    )

    if col_jeunes and col_adultes and col_vieux:
        line_plot_interactive(
            dff,
            [col_jeunes, col_adultes, col_vieux],
            "Comparaison du chômage — jeunes vs adultes vs personnes âgées",
            y_label="Taux (%)",
        )
        st.write(
            "Cet indicateur compare l’évolution du chômage des jeunes et des adultes afin de mettre en évidence "
            "les populations les plus sensibles aux fluctuations économiques, notamment lorsque les jeunes sont "
            "plus exposés aux crises et à la précarité du marché du travail."
        )
    else:
        st.info(
            "Colonnes jeunes/adultes introuvables (attendues: taux_chomage_15_24_insee et taux_chomage_25_49_insee, ou équivalents)."
        )

    st.divider()


# =========================================================
# PAGE EDA (VIDE)
# =========================================================
elif page == "📊 EDA":
    st.title("📊 EDA")
    st.write("On laissera cette partie vide pour l’instant, on la fera ensuite.")


# =========================================================
# FEATURE IMPORTANCE (score combiné)
# =========================================================
elif page == "🧠 Feature Importance":
    st.title("🧠 Variables les plus explicatives du chômage (MI + Lasso)")

    X_raw, y = prepare_xy(df.dropna(subset=[TARGET]))
    X = clean_numeric(X_raw).select_dtypes(include=[np.number])

    if X.shape[1] == 0:
        st.error("Aucune variable numérique disponible.")
        st.stop()

    # MI
    X_imp = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    y_imp = y.fillna(y.median())
    mi = mutual_info_regression(X_imp, y_imp, random_state=42)
    mi_rank = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    # Lasso
    tscv = TimeSeriesSplit(n_splits=5)
    lasso_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=tscv, random_state=42, max_iter=20000)),
        ]
    )
    lasso_pipe.fit(X, y)
    lasso_coef = pd.Series(lasso_pipe.named_steps["lasso"].coef_, index=X.columns)
    lasso_rank = lasso_coef.abs().sort_values(ascending=False)

    # enlever taux_chomage_ocde
    mi_rank_clean = mi_rank.drop("taux_chomage_ocde", errors="ignore")
    lasso_rank_clean = lasso_rank.drop("taux_chomage_ocde", errors="ignore")

    def minmax_norm(s: pd.Series) -> pd.Series:
        s = s.astype(float).copy()
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return s
        s_min, s_max = s.min(), s.max()
        if np.isclose(s_max - s_min, 0.0):
            return pd.Series(0.0, index=s.index)
        return (s - s_min) / (s_max - s_min)

    mi_norm = minmax_norm(mi_rank_clean)
    lasso_norm = minmax_norm(lasso_rank_clean)

    common_vars = mi_norm.index.intersection(lasso_norm.index)

    combined = pd.DataFrame(
        {"MI_norm": mi_norm.loc[common_vars], "Lasso_norm": lasso_norm.loc[common_vars]}
    )
    combined["Score_combine"] = combined.mean(axis=1)

    TOP_N = st.slider("Nombre de variables à afficher (Top N)", 5, 30, 10)

    top_combined = combined.sort_values("Score_combine", ascending=False).head(TOP_N)
    top_plot = top_combined.sort_values("Score_combine")

    st.subheader("Top variables (score combiné)")
    st.dataframe(top_combined, use_container_width=True)

    fig = plt.figure(figsize=(9, 6))
    plt.barh(top_plot.index, top_plot["Score_combine"], color="steelblue")
    plt.title("Variables les plus explicatives du chômage\n(score combiné MI + Lasso)")
    plt.xlabel("Score combiné")
    plt.ylabel("Variables")
    plt.tight_layout()
    st.pyplot(fig)

    st.write(
        "Cet indicateur combine Mutual Information (capte aussi des liens non-linéaires) et Lasso "
        "(sélection linéaire parcimonieuse) afin d’obtenir un classement unique des variables "
        "les plus explicatives du taux de chômage."
    )


# =========================================================
# PREDICTION
# =========================================================
elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction du chômage (baseline)")

    dff = df.dropna(subset=[DATE_COL, TARGET]).sort_values(DATE_COL).copy()
    X_raw, y = prepare_xy(dff)

    X = clean_numeric(X_raw).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.shape[1] == 0:
        st.error("Aucune variable numérique disponible pour l'entraînement.")
        st.stop()

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.3f}")
    m2.metric("RMSE", f"{rmse:.3f}")
    m3.metric("R²", f"{r2:.3f}")
