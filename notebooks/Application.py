import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import scipy.stats as stats
import seaborn as sns
import scipy.stats as stats

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="Chômage France", layout="wide")

DEFAULT_CSV_PATH = "../data/processed/02_dataset_silver.csv"
TARGET = "taux_chomage_total_insee"
DATE_COL = "date"
DROP_COLS_ALWAYS = ["taux_chomage_ocde"]


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
    d = df.copy()
    num_cols = d.select_dtypes(include=[np.number]).columns
    d[num_cols] = d[num_cols].apply(pd.to_numeric, errors="coerce")
    d[num_cols] = d[num_cols].replace([np.inf, -np.inf], np.nan)
    max_float = np.finfo(np.float64).max
    d[num_cols] = d[num_cols].mask(d[num_cols].abs() > max_float, np.nan)
    return d


def find_age_cols(df: pd.DataFrame):
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
    fig = px.line(
        dff,
        x=DATE_COL,
        y=y_cols,
        title=title,
        labels={"value": y_label, "variable": "Série"},
    )
    st.plotly_chart(fig, use_container_width=True)


def create_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET in df.columns:
        df["chomage_lag1"] = df[TARGET].shift(1)

    if "pib_logdiff" in df.columns:
        df["pib_logdiff_lag1"] = df["pib_logdiff"].shift(1)

    if "nb_interimaires_var" in df.columns:
        df["nb_interimaires_var_lag1"] = df["nb_interimaires_var"].shift(1)

    if "indicateur_retournement_conjoncturel" in df.columns:
        df["indicateur_retournement_conjoncturel_lag1"] = df[
            "indicateur_retournement_conjoncturel"
        ].shift(1)

    if "population_active_var" in df.columns:
        df["population_active_var_lag1"] = df["population_active_var"].shift(1)

    if DATE_COL in df.columns:
        df["is_covid"] = (df[DATE_COL].dt.year == 2020).astype(int)

    return df


def plot_histogram(df: pd.DataFrame, col: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[col].dropna(), bins=30, kde=True, color="lightblue", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Fréquence")
    st.pyplot(fig)


def plot_boxplot(df: pd.DataFrame, col: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.boxplot(x=df[col].dropna(), color="lightgreen", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col)
    st.pyplot(fig)


def plot_violin(df: pd.DataFrame, col: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.violinplot(x=df[col].dropna(), color="plum", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col)
    st.pyplot(fig)


def plot_qq(df: pd.DataFrame, col: str, title: str):
    fig = plt.figure(figsize=(6, 4))
    stats.probplot(df[col].dropna(), dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)


def plot_corr_heatmap(df: pd.DataFrame, target: str, top_n: int = 13):
    num_df = df.select_dtypes(include=[np.number]).copy()
    num_df = num_df.replace([np.inf, -np.inf], np.nan)

    corr = num_df.corr(numeric_only=True)
    corr_target = (
        corr[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
    )

    top_vars = corr_target.head(top_n).index.tolist()
    corr_subset = num_df[top_vars + [target]].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_subset, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Heatmap des corrélations avec la cible")
    st.pyplot(fig)

    return corr_target


# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("Chômage France 🇫🇷")
st.sidebar.caption("Évolution du chômage en France")
csv_path = DEFAULT_CSV_PATH

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "📊 EDA", "📊 Dashboard", "🔮 Prédiction"],
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
            "Date fin",
            value=max_d,
            min_value=min_d,
            max_value=max_d,
            key="dash_end",
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

    st.subheader("1) Évolution du taux de chômage (global)")
    line_plot_interactive(
        dff, TARGET, "Évolution du taux de chômage total (INSEE)", y_label="Taux (%)"
    )

    st.divider()

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

    st.divider()

    st.subheader("3) Chômage — Tranches d'âges")
    col_jeunes = pick_first_existing(
        df, ["taux_chomage_15_24_insee", "taux_chomage_moins25_insee"]
    )
    col_adultes = pick_first_existing(
        df, ["taux_chomage_25_49_insee", "taux_chomage_25_54_insee"]
    )
    col_vieux = pick_first_existing(
        df,
        [
            "taux_chomage_50_plus_insee",
            "taux_chomage_50_64_insee",
            "taux_chomage_55_64_insee",
        ],
    )

    available_age_compare = [
        c for c in [col_jeunes, col_adultes, col_vieux] if c is not None
    ]

    if len(available_age_compare) >= 2:
        line_plot_interactive(
            dff,
            available_age_compare,
            "Comparaison du chômage — jeunes, adultes et seniors",
            y_label="Taux (%)",
        )


# =========================================================
# PAGE EDA
# =========================================================
elif page == "📊 EDA":
    st.title("📊 EDA — Analyse exploratoire des données")

    st.divider()

    # =========================
    # 1. Taux de chômage total
    # =========================
    st.subheader("1) Taux de chômage total")
    plot_histogram(
        df, "taux_chomage_total_insee", "Histogramme — taux de chômage total"
    )
    st.write(
        "La distribution du taux de chômage total est relativement concentrée, ce qui suggère une certaine stabilité "
        "du chômage global sur la période étudiée, avec des fluctuations mais sans dispersion extrême."
    )

    plot_boxplot(df, "taux_chomage_total_insee", "Boxplot — taux de chômage total")
    st.write(
        "Le boxplot permet de vérifier la dispersion et la présence éventuelle de valeurs atypiques. "
        "il confirme surtout un étalement modéré autour de la médiane."
    )

    plot_qq(df, "taux_chomage_total_insee", "Q-Q Plot — taux de chômage total")
    st.write(
        "Le Q-Q plot sert à vérifier si la distribution du taux de chômage total se rapproche d’une loi normale. "
        "Cela aide à juger si certaines méthodes statistiques linéaires sont adaptées ou non."
    )

    st.divider()

    # =========================
    # 2. Taux de chômage femmes
    # =========================
    plot_histogram(
        df, "taux_chomage_femme_insee", "Histogramme — taux de chômage féminin"
    )
    st.write(
        "La distribution du chômage féminin apparaît proche de celle du chômage total, ce qui est cohérent avec "
        "la forte corrélation observée entre cette variable et la cible."
    )

    plot_boxplot(df, "taux_chomage_femme_insee", "Boxplot — taux de chômage féminin")
    st.write(
        "Le boxplot met en évidence la variabilité du chômage féminin et permet de comparer sa dispersion "
        "à celle du chômage total dans une logique de lecture socio-économique."
    )

    plot_qq(df, "taux_chomage_femme_insee", "Q-Q Plot — taux de chômage féminin")
    st.write(
        "Ce graphique permet d’évaluer la normalité approximative de la variable, ce qui est utile avant "
        "toute modélisation ou analyse paramétrique."
    )

    st.divider()

    # =========================
    # 3. Climat des affaires
    # =========================
    if "indicateur_climat_affaires" in df.columns:
        st.subheader("3) Climat des affaires")
        plot_histogram(
            df, "indicateur_climat_affaires", "Histogramme — climat des affaires"
        )
        st.write(
            "Le climat des affaires est un indicateur conjoncturel important : son histogramme permet de voir "
            "si les observations sont réparties de manière homogène ou concentrées autour de certaines phases économiques."
        )

        plot_boxplot(df, "indicateur_climat_affaires", "Boxplot — climat des affaires")
        st.write(
            "Le boxplot du climat des affaires aide à repérer les périodes atypiques, notamment celles pouvant "
            "correspondre à des chocs économiques ou à des retournements de conjoncture."
        )

        plot_qq(df, "indicateur_climat_affaires", "Q-Q Plot — climat des affaires")
        st.write(
            "Le Q-Q plot permet ici de vérifier si la distribution du climat des affaires s’écarte fortement "
            "de la normalité, ce qui peut orienter le choix du modèle ensuite."
        )

        st.divider()

    # =========================
    # 4. IPC
    # =========================
    if "ipc" in df.columns:
        st.subheader("4) IPC")
        plot_histogram(df, "ipc", "Histogramme — IPC")
        st.write(
            "L’IPC présente  une corrélation négative notable avec le taux de chômage, "
            "ce qui en fait une variable macroéconomique intéressante à explorer plus finement."
        )
        plot_boxplot(df, "ipc", "Boxplot — IPC")
        st.write(
            "Le boxplot de l’IPC permet de vérifier si certaines périodes inflationnistes ou désinflationnistes "
            "se démarquent nettement du reste des observations."
        )

        plot_qq(df, "ipc", "Q-Q Plot — IPC")
        st.write(
            "Le Q-Q plot aide à juger si l’IPC suit une distribution proche de la normale ou s’il existe "
            "des asymétries importantes dans la série."
        )

        st.divider()
    # =========================
    # 5. Demandeurs d'emploi total
    # =========================
    if "demandeur_total_abcd_moins25" in df.columns:
        st.subheader("5) Demandeurs d’emploi total (moins de 25 ans)")
        plot_histogram(
            df,
            "demandeur_total_abcd_moins25",
            "Histogramme — demandeurs d’emploi total (moins de 25 ans)",
        )
        st.write(
            "Le volume de demandeurs d’emploi de moins de 25 ans est particulièrement pertinent car il est "
            "positivement corrélé à la cible dans le notebook, ce qui confirme la vulnérabilité des jeunes au chômage."
        )

        plot_violin(
            df,
            "demandeur_total_abcd_moins25",
            "Violin plot — demandeurs d’emploi total (moins de 25 ans)",
        )
        st.write(
            "Le violin plot complète l’histogramme en montrant mieux la densité de la distribution et les zones "
            "où les observations sont les plus concentrées."
        )

        plot_qq(
            df,
            "demandeur_total_abcd_moins25",
            "Q-Q Plot — demandeurs d’emploi total (moins de 25 ans)",
        )
        st.write(
            "Le Q-Q plot permet d’apprécier les écarts éventuels à la normalité, souvent fréquents sur des séries de volumes économiques."
        )

        st.divider()

    # =========================
    # 6. MRO
    # =========================
    if "mro" in df.columns:
        st.subheader("6) MRO")
        plot_histogram(df, "mro", "Histogramme — MRO")
        st.write(
            "Le taux MRO présente dans le notebook une corrélation négative avec le chômage. "
            "Son histogramme permet de visualiser la structure générale de cette variable monétaire."
        )

        plot_violin(df, "mro", "Violin plot — MRO")
        st.write(
            "Le violin plot met en évidence la densité des niveaux de MRO observés et aide à repérer "
            "si certaines valeurs dominent la série."
        )

        plot_qq(df, "mro", "Q-Q Plot — MRO")
        st.write(
            "Le Q-Q plot permet ici d’évaluer dans quelle mesure la variable suit une forme gaussienne, "
            "ce qui peut influencer le comportement des modèles linéaires."
        )

        st.divider()

    # =========================
    # 7. Heatmap corrélations
    # =========================
    st.subheader("7) Heatmap des corrélations les plus fortes")
    corr_target = plot_corr_heatmap(df, TARGET, top_n=13)

    st.write(
        "La heatmap confirme que les variables les plus liées au taux de chômage total sont surtout "
        "les autres taux de chômage par sous-populations, en particulier les 25–49 ans, les hommes, "
        "les femmes et les jeunes. On observe aussi des relations notables avec l’IPC, le MRO "
        "et certains indicateurs du marché du travail."
    )

    st.subheader("Top corrélations avec la cible")
    st.dataframe(corr_target.head(15).to_frame("corrélation"), use_container_width=True)


# =========================================================
# PREDICTION — reprise fidèle du notebook
# =========================================================
elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction du chômage")

    df_pred = create_prediction_features(df)

    final_features = [
        "chomage_lag1",
        "pib_logdiff_lag1",
        "nb_interimaires_var_lag1",
        "indicateur_retournement_conjoncturel_lag1",
        "population_active_var_lag1",
        "is_covid",
    ]
    target = TARGET

    existing_features = [c for c in final_features if c in df_pred.columns]
    missing_features = [c for c in final_features if c not in df_pred.columns]

    if missing_features:
        st.warning(f"Variables absentes dans le dataset : {missing_features}")

    if len(existing_features) == 0:
        st.error("Aucune feature de prédiction disponible.")
        st.stop()

    final_features = existing_features

    # -----------------------------------------------------
    # 1. Entraînement du modèle de production + coefficients
    # -----------------------------------------------------
    st.subheader("1) Modèle Ridge de production")

    df_final = df_pred.dropna(subset=final_features + [target]).copy()

    model_prod = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model_prod.fit(df_final[final_features], df_final[target])

    coefs = model_prod.named_steps["ridge"].coef_
    coef_df = pd.DataFrame(
        {"Variable": final_features, "Coefficient": coefs}
    ).sort_values("Coefficient", key=np.abs, ascending=False)

    st.dataframe(coef_df, use_container_width=True)

    # -----------------------------------------------------
    # 2. Projection 3 scénarios
    # -----------------------------------------------------
    st.subheader("2) Projection du chômage — scénarios")

    last_date = df_pred[DATE_COL].iloc[-1]
    start_rate = df_pred[target].iloc[-1]
    dates_projection = pd.date_range(start=last_date, periods=7, freq="QS")

    def simuler_2026_fixed(hypo_base, start_val, model, features):
        results = [start_val]
        current_val = start_val

        for _ in range(6):
            X = pd.DataFrame([hypo_base])
            X["chomage_lag1"] = current_val
            X = X[features]
            pred = model.predict(X)[0]
            results.append(pred)
            current_val = pred
        return results

    moyennes_2025 = {f: df_final[f].tail(4).mean() for f in final_features}

    hypo_opt = {
        "chomage_lag1": start_rate,
        "pib_logdiff_lag1": 0.012 if "pib_logdiff_lag1" in final_features else 0,
        "nb_interimaires_var_lag1": (
            -0.02 if "nb_interimaires_var_lag1" in final_features else 0
        ),
        "indicateur_retournement_conjoncturel_lag1": (
            1.0 if "indicateur_retournement_conjoncturel_lag1" in final_features else 0
        ),
        "population_active_var_lag1": (
            -0.001 if "population_active_var_lag1" in final_features else 0
        ),
        "is_covid": 0,
    }

    hypo_middle = moyennes_2025.copy()
    hypo_middle["chomage_lag1"] = start_rate
    hypo_middle["is_covid"] = 0

    hypo_pess = {
        "chomage_lag1": start_rate,
        "pib_logdiff_lag1": -0.003 if "pib_logdiff_lag1" in final_features else 0,
        "nb_interimaires_var_lag1": (
            0.03 if "nb_interimaires_var_lag1" in final_features else 0
        ),
        "indicateur_retournement_conjoncturel_lag1": (
            -1.5 if "indicateur_retournement_conjoncturel_lag1" in final_features else 0
        ),
        "population_active_var_lag1": (
            0.004 if "population_active_var_lag1" in final_features else 0
        ),
        "is_covid": 0,
    }

    for d in [hypo_opt, hypo_middle, hypo_pess]:
        for f in final_features:
            if f not in d:
                d[f] = 0

    curve_opt = simuler_2026_fixed(hypo_opt, start_rate, model_prod, final_features)
    curve_mid = simuler_2026_fixed(hypo_middle, start_rate, model_prod, final_features)
    curve_pess = simuler_2026_fixed(hypo_pess, start_rate, model_prod, final_features)

    fig = plt.figure(figsize=(14, 7))
    plt.plot(
        df_pred[DATE_COL].tail(20),
        df_pred[target].tail(20),
        label="Historique (INSEE)",
        color="black",
        linewidth=3,
    )
    plt.plot(
        dates_projection,
        curve_opt,
        "g--o",
        label="Scénario : Relance Économique",
        alpha=0.8,
    )
    plt.plot(
        dates_projection,
        curve_mid,
        "b-d",
        label="Scénario : Dynamique 2025",
        linewidth=2,
    )
    plt.plot(
        dates_projection,
        curve_pess,
        "r--s",
        label="Scénario : Récession & Tension",
        alpha=0.8,
    )
    plt.scatter(last_date, start_rate, color="black", s=100, zorder=5)
    plt.title(
        "Projection du Chômage en France (Horizon 2027)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.ylabel("Taux de chômage (%)", fontsize=12)
    plt.axvline(x=last_date, color="gray", linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", frameon=True, shadow=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------------------------------
    # 3. Backtesting post-covid
    # -----------------------------------------------------
    st.subheader("3) Backtesting : période post-covid")

    df_test = df_pred.dropna(subset=final_features + [target]).copy()
    split_idx = int(len(df_test) * 0.82)

    train = df_test.iloc[:split_idx]
    test = df_test.iloc[split_idx:]

    model_backtest = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
    )
    model_backtest.fit(train[final_features], train[target])

    predictions = model_backtest.predict(test[final_features])

    mae = mean_absolute_error(test[target], predictions)
    mse = mean_squared_error(test[target], predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test[target], predictions)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("MSE", f"{mse:.3f}")
    c3.metric("RMSE", f"{rmse:.3f}")
    c4.metric("R²", f"{r2:.3f}")

    fig = plt.figure(figsize=(14, 7))
    plt.plot(
        df_test[DATE_COL],
        df_test[target],
        label="Données Réelles (INSEE)",
        color="black",
        alpha=0.3,
    )
    plt.plot(
        test[DATE_COL],
        test[target],
        label="Réel (Zone de Test)",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        test[DATE_COL],
        predictions,
        label="Prédiction du Modèle",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.title(
        "Backtesting : Comparaison entre le Chômage Réel et les Prédictions du Modèle",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Taux de chômage (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    st.pyplot(fig)

    # -----------------------------------------------------
    # 4. Validation croisée TimeSeriesSplit
    # -----------------------------------------------------
    st.subheader("4) Validation croisée chronologique")

    tscv = TimeSeriesSplit(n_splits=5)
    scores_mae = []
    scores_r2 = []
    rows = []

    X_ml = df_final[final_features].values
    y_ml = df_final[target].values

    for i, (train_index, test_index) in enumerate(tscv.split(X_ml)):
        X_train, X_test = X_ml[train_index], X_ml[test_index]
        y_train, y_test_cv = y_ml[train_index], y_ml[test_index]

        model_prod.fit(X_train, y_train)
        y_pred = model_prod.predict(X_test)

        mae_cv = mean_absolute_error(y_test_cv, y_pred)
        r2_cv = r2_score(y_test_cv, y_pred)

        scores_mae.append(mae_cv)
        scores_r2.append(r2_cv)

        rows.append(
            {
                "Fold": i + 1,
                "Taille Train": len(train_index),
                "Taille Test": len(test_index),
                "MAE": round(mae_cv, 4),
                "R²": round(r2_cv, 4),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.write(f"Erreur Moyenne Globale (MAE) : {np.mean(scores_mae):.4f}")
    st.write(f"Écart-type des erreurs : {np.std(scores_mae):.4f}")
    st.write(
        'Le modèle devient de plus en plus précis (MAE qui baisse), mais il semble "moins bon" statistiquement (R² qui baisse).'
    )
    st.write(
        "Fold 4 (MAE 0.33) :  \"C'est ici que ton modèle \"souffre\" le plus. Chronologiquement, ce pli correspond probablement à la période COVID-19 ou à la remontée brutale de l'inflation (2022). Le modèle, qui repose sur des relations linéaires, a été surpris par une cassure nette qu'il n'avait pas vue venir.\""
    )
    st.write(
        '- Fold 5 (MAE 0.14) : "C\'est ton pli le plus récent. Une MAE de 0.14 est incroyable pour un économiste. Cela prouve que sur la dynamique actuelle (post-COVID), ton modèle est parfaitement "calé"."'
    )
    st.write(
        '- Fold 3 (MAE 0.15) : "Période de croissance régulière où les variables (PIB, intérim) expliquaient très bien le chômage."'
    )

    # -----------------------------------------------------
    # 5. Diagnostic des résidus
    # -----------------------------------------------------
    st.subheader("5) Diagnostic des résidus")

    y_test_pred = model_prod.predict(df_test[final_features])
    residuals = df_test[target] - y_test_pred

    fig = plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, color="purple", bins=15)
    plt.axvline(x=0, color="red", linestyle="--")
    plt.title("Distribution des Résidus (Erreurs)", fontsize=13)
    plt.xlabel("Écart (Réel - Prédit) en points de %")
    plt.ylabel("Fréquence")

    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot : Alignement sur la Loi Normale", fontsize=13)

    plt.tight_layout()
    st.pyplot(fig)

    biais_moyen = np.mean(residuals)
    st.write(f"Biais moyen du modèle : {biais_moyen:.6f}")

    # -----------------------------------------------------
    # 6. Duel de modèles
    # -----------------------------------------------------
    st.subheader("6) Duel de modèles")

    models = {
        "Ridge (Linéaire)": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "Random Forest (Non-Linéaire)": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
        "Gradient Boosting (Non-Linéaire)": HistGradientBoostingRegressor(
            random_state=42
        ),
    }

    rows_models = []
    results_comparison = {}

    for name, model in models.items():
        model.fit(train[final_features], train[target])
        preds = model.predict(test[final_features])

        mae_model = mean_absolute_error(test[target], preds)
        r2_model = r2_score(test[target], preds)

        results_comparison[name] = preds
        rows_models.append(
            {
                "Modèle": name,
                "MAE": round(mae_model, 4),
                "R²": round(r2_model, 4),
            }
        )

    st.dataframe(pd.DataFrame(rows_models), use_container_width=True)

    fig = plt.figure(figsize=(14, 7))
    plt.plot(
        test[DATE_COL], test[target], label="Réel (INSEE)", color="black", linewidth=3
    )

    for name, preds in results_comparison.items():
        plt.plot(test[DATE_COL], preds, label=f"Prédit par {name}", linestyle="--")

    plt.title("Duel de modèles : Lequel suit le mieux la réalité ?", fontsize=14)
    plt.legend()
    st.pyplot(fig)
    st.write(
        "Nous avons comparé notre approche linéaire (Ridge) à des algorithmes de pointe non-linéaires (Random Forest, Gradient Boosting). Le Ridge s'est révélé bien plus performant, car la dynamique du chômage est fortement marquée par une inertie linéaire que les modèles complexes tendent à sur-interpréter (overfitting), surtout sur un échantillon de données macroéconomiques réduit."
    )
