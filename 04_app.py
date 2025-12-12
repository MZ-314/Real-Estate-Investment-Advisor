import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from difflib import get_close_matches
import plotly.express as px

ROOT = Path(__file__).parent

DATA_PATH = ROOT / "data" / "cleaned_housing_data.csv"
CLF_PATH  = ROOT / "models" / "investment_classifier_memsafe.pkl"
REG_PATH  = ROOT / "models" / "future_price_regressor_memsafe.pkl"

st.set_page_config(page_title="Real Estate Dashboard", layout="wide", initial_sidebar_state="expanded")

PALETTE = ["#2b8cbe", "#f03b20", "#7fc97f", "#984ea3", "#ff7f00", "#6a3d9a"]


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    for c in ["City", "Locality", "State"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "Price_per_SqFt" not in df.columns and "Price_in_Lakhs" in df.columns and "Size_in_SqFt" in df.columns:
        df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000.0) / df["Size_in_SqFt"].replace(0, np.nan)
    return df

@st.cache_resource(show_spinner=False)
def load_models():
    clf = joblib.load(CLF_PATH)
    reg = joblib.load(REG_PATH)
    return clf, reg

@st.cache_data(show_spinner=False)
def build_freq_maps(df, high_card_cols):
    freq_maps = {}
    for c in high_card_cols:
        freq_maps[c] = df[c].value_counts(dropna=False).to_dict()
    return freq_maps

def apply_freq_encoding(row_df, freq_maps):
    total_counts_cache = {}
    for col, fmap in freq_maps.items():
        fname = f"{col}_freq"
        fname_ratio = f"{col}_freq_ratio"
        val = row_df.iloc[0].get(col, np.nan)
        cnt = fmap.get(val, 0)
        row_df[fname] = cnt
        total = total_counts_cache.get(col)
        if total is None:
            total = float(sum(fmap.values()))
            total_counts_cache[col] = total
        row_df[fname_ratio] = cnt / total if total > 0 else 0.0
    return row_df

def compute_price_per_sqft(price_lakhs, size_sqft):
    try:
        return (float(price_lakhs) * 100000.0) / float(size_sqft)
    except Exception:
        return np.nan

def find_best_city_match(city_name, available_cities):
    if city_name in available_cities:
        return city_name
    ci_map = {c.lower(): c for c in available_cities}
    if city_name.lower() in ci_map:
        return ci_map[city_name.lower()]
    matches = get_close_matches(city_name, available_cities, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_model_input_columns(model):
    """
    Try to extract the feature names that the model/pipeline expects.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "steps"):
        for name, step in model.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return []
df = load_data()
clf, reg = load_models()

CARD_THRESHOLD = 30
all_cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
high_card_cols = [c for c in all_cat_cols if df[c].nunique() > CARD_THRESHOLD]
low_card_cols = [c for c in all_cat_cols if df[c].nunique() <= CARD_THRESHOLD]
freq_maps = build_freq_maps(df, high_card_cols)

st.sidebar.title("Property Input")
st.sidebar.markdown("Enter basics and click **Predict**")

state_list = sorted(df["State"].dropna().unique().tolist())
if len(state_list) == 0:
    state = st.sidebar.selectbox("State", ["No states available"])
else:
    state = st.sidebar.selectbox("State", state_list)

city_list = sorted(df[df["State"] == state]["City"].dropna().unique().tolist())
if len(city_list) == 0:
    city = st.sidebar.selectbox("City", ["No cities available"])
else:
    city = st.sidebar.selectbox("City", city_list)

locality_list = sorted(df[df["City"] == city]["Locality"].dropna().unique().tolist())
if len(locality_list) == 0:
    locality = st.sidebar.selectbox("Locality", ["No localities available"])
else:
    locality = st.sidebar.selectbox("Locality", locality_list)

property_type = st.sidebar.selectbox("Property Type", sorted(df["Property_Type"].dropna().unique().tolist()))
bhk = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=3)
size_sqft = st.sidebar.number_input("Size (SqFt)", min_value=200, max_value=20000, value=1200)
price_lakhs = st.sidebar.number_input("Current Price (Lakhs)", min_value=0.1, value=80.0, step=0.5)
furnished = st.sidebar.selectbox("Furnished", sorted(df["Furnished_Status"].dropna().unique().tolist()))
age = st.sidebar.number_input("Age (years)", min_value=0, max_value=200, value=5)
nearby_schools = st.sidebar.number_input("Nearby Schools", min_value=0, max_value=100, value=3)
public_transport = st.sidebar.selectbox("Public Transport", sorted(df["Public_Transport_Accessibility"].dropna().unique().tolist()))
parking = st.sidebar.selectbox("Parking", sorted(df["Parking_Space"].dropna().unique().tolist()))
amenities = st.sidebar.text_input("Amenities (comma-separated)", value="Pool,Park")

if st.sidebar.button("Predict"):
    input_row = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Price_in_Lakhs": price_lakhs,
        "Price_per_SqFt": compute_price_per_sqft(price_lakhs, size_sqft),
        "Furnished_Status": furnished,
        "Age_of_Property": age,
        "Nearby_Schools": nearby_schools,
        "Public_Transport_Accessibility": public_transport,
        "Parking_Space": parking,
        "Amenities": amenities
    }

    dataset_cols = df.columns.tolist()
    for c in dataset_cols:
        if c not in input_row and c not in ["Good_Investment", "Future_Price_5Y", "City_Median_Price_per_SqFt"]:
            if df[c].dtype.kind in "biufc":
                input_row[c] = 0
            else:
                try:
                    input_row[c] = df[c].mode().iloc[0] if not df[c].mode().empty else ""
                except Exception:
                    input_row[c] = ""

    input_df = pd.DataFrame([input_row])

    num_cols_train = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for c in num_cols_train:
        if c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c], errors="coerce").fillna(0)

    input_df = apply_freq_encoding(input_df, freq_maps)

    available_cities = sorted(df["City"].dropna().unique().tolist())
    matched_city = find_best_city_match(city, available_cities)
    if matched_city:
        city_med = df.loc[df["City"] == matched_city, "Price_per_SqFt"].median()
        if pd.isna(city_med):
            city_med = df["Price_per_SqFt"].median()
        input_df["City_Median_Price_per_SqFt"] = city_med
    else:
        input_df["City_Median_Price_per_SqFt"] = df["Price_per_SqFt"].median()

    model_features = get_model_input_columns(clf)
    if not model_features:
        derived_freq_cols = []
        for c in high_card_cols:
            derived_freq_cols += [f"{c}_freq", f"{c}_freq_ratio"]
        model_features = list(dict.fromkeys(dataset_cols + derived_freq_cols))

    final_input = pd.DataFrame(columns=model_features)
    for col in model_features:
        if col in input_df.columns:
            final_input.loc[0, col] = input_df.loc[0, col]
        else:
            if col.endswith("_freq") or col.endswith("_freq_ratio"):
                final_input.loc[0, col] = 0
            elif col in df.columns:
                if df[col].dtype.kind in "biufc":
                    try:
                        final_input.loc[0, col] = float(df[col].mean())
                    except Exception:
                        final_input.loc[0, col] = 0
                else:
                    try:
                        final_input.loc[0, col] = df[col].mode().iloc[0] if not df[col].mode().empty else ""
                    except Exception:
                        final_input.loc[0, col] = ""
            else:
                final_input.loc[0, col] = 0
    for c in final_input.columns:
        if c in num_cols_train:
            final_input[c] = pd.to_numeric(final_input[c], errors="coerce").fillna(0)

    try:
        proba = clf.predict_proba(final_input)[0][1] if hasattr(clf, "predict_proba") else None
        pred_class = int(clf.predict(final_input)[0])
    except Exception as e:
        st.error("Classifier failed — column mismatch or model error. See exception below.")
        st.exception(e)
        st.stop()

    try:
        reg_features = get_model_input_columns(reg)
        use_for_reg = final_input
        if reg_features:
            tmp = pd.DataFrame(index=[0], columns=reg_features)
            for c in reg_features:
                if c in final_input.columns:
                    tmp.loc[0, c] = final_input.loc[0, c]
                else:
                    tmp.loc[0, c] = 0
            for c in reg_features:
                if c in num_cols_train:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)
            use_for_reg = tmp
        pred_price = reg.predict(use_for_reg)[0]
    except Exception as e:
        st.error("Regressor failed — column mismatch or model error. See exception below.")
        st.exception(e)
        st.stop()

    k1, k2, k3 = st.columns([1,1,1])
    with k1:
        st.metric("Current Price", f"₹ {price_lakhs:,.2f} Lakhs")
    with k2:
        st.metric("Predicted Price (5Y)", f"₹ {pred_price:,.2f} Lakhs")
    with k3:
        score = round(proba*100,2) if proba is not None else "N/A"
        label = "Good" if pred_class==1 else "Not Good"
        st.metric("Investment Score", f"{score}%", delta=label)

    col_left, col_right = st.columns([1.3,1])
    with col_left:
        st.subheader("Recommendation")
        if pred_class == 1:
            st.success(f"✅ Recommended as GOOD investment (confidence {proba:.2f})")
        else:
            st.warning(f"⚠️ Not recommended (confidence {proba:.2f})")
        st.write("Summary:")
        st.write(f"- State: {state}  •  City: {city}  •  Locality: {locality}")
        st.write(f"- BHK: {bhk}, Size: {size_sqft} sqft, Furnished: {furnished}")
        st.write(f"- Nearby Schools: {nearby_schools}, Public Transport: {public_transport}")
        if st.button("Save Snapshot"):
            out_dir = ROOT / "predictions"
            out_dir.mkdir(parents=True, exist_ok=True)
            fn = out_dir / f"dashboard_pred_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_df = final_input.copy()
            save_df["Pred_Good_Investment"] = pred_class
            save_df["Pred_5Y_Price_Lakhs"] = pred_price
            save_df.to_csv(fn, index=False)
            st.success(f"Saved to {fn}")

    with col_right:
        st.subheader("Price Comparison")
        pc_df = pd.DataFrame({
            "Scenario": ["Now", "5 Years"],
            "Price_Lakhs": [price_lakhs, pred_price]
        })
        fig_pc = px.bar(pc_df, x="Scenario", y="Price_Lakhs", text="Price_Lakhs",
                        color="Scenario", color_discrete_sequence=PALETTE[:2],
                        title="Now vs 5Y Prediction")
        fig_pc.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_pc.update_layout(yaxis_title="Price (Lakhs)", uniformtext_minsize=8, uniformtext_mode='hide', height=360)
        st.plotly_chart(fig_pc, use_container_width=True)


    st.markdown("---")
    st.subheader("Local Market Snapshot")

    if matched_city:
        local_stats = df[df["City"] == matched_city].copy()
    else:
        local_stats = pd.DataFrame()

    if not local_stats.empty:
        avg_psf = local_stats["Price_per_SqFt"].mean()
        med_price = local_stats["Price_in_Lakhs"].median()
        st.write(f"- Showing data for city: **{matched_city}** (matched from input '{city}')")
        st.write(f"- Avg Price per SqFt (city): ₹ {avg_psf:.2f}")
        st.write(f"- Median Property Price (city): ₹ {med_price:.2f} Lakhs")

        st.dataframe(local_stats[["Locality","Price_in_Lakhs","Price_per_SqFt","BHK"]]
                     .sort_values("Price_per_SqFt", ascending=False).head(10), height=220)

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.caption("Price per SqFt distribution (city)")
            fig1 = px.histogram(local_stats, x="Price_per_SqFt", nbins=30, title="Price per SqFt",
                                color_discrete_sequence=[PALETTE[0]])
            fig1.update_layout(xaxis_title="Price per SqFt", yaxis_title="Count", height=360)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            st.caption("BHK distribution (city)")
            # robust BHK counts (use groupby.size -> deterministic columns)
            if "BHK" in local_stats.columns:
                bhk_counts = local_stats.groupby("BHK").size().reset_index(name="Count")
                bhk_counts["BHK"] = bhk_counts["BHK"].astype(str)
            else:
                bhk_counts = pd.DataFrame({"BHK": [], "Count": []})
            # color by BHK so categories receive palette colors
            fig2 = px.bar(bhk_counts, x="BHK", y="Count", color="BHK", title="BHK distribution",
                          color_discrete_sequence=PALETTE)
            fig2.update_layout(xaxis_title="BHK", yaxis_title="Count", height=360, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        with c3:
            st.caption("Price vs Size scatter (city)")
            # color by BHK to show category colors on scatter markers
            if "BHK" in local_stats.columns:
                local_stats["BHK_str"] = local_stats["BHK"].astype(str)
                fig3 = px.scatter(local_stats, x="Size_in_SqFt", y="Price_in_Lakhs",
                                  color="BHK_str", color_discrete_sequence=PALETTE,
                                  hover_data=["Locality","BHK"], title="Price vs Size")
                # hide legend title
                fig3.update_layout(legend_title_text="BHK", height=360)
            else:
                fig3 = px.scatter(local_stats, x="Size_in_SqFt", y="Price_in_Lakhs",
                                  hover_data=["Locality"], title="Price vs Size")
                fig3.update_layout(height=360)
            fig3.update_layout(xaxis_title="Size (SqFt)", yaxis_title="Price (Lakhs)")
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No local data for this city in training dataset.")
        suggestions = get_close_matches(city, available_cities, n=5, cutoff=0.5)
        if suggestions:
            st.write("Did you mean one of these cities from training data?")
            st.write(", ".join(suggestions))
        st.write("Showing overall dataset-level summary (fallback):")
        overall_avg_psf = df["Price_per_SqFt"].mean()
        overall_med_price = df["Price_in_Lakhs"].median()
        st.write(f"- Avg Price per SqFt (dataset): ₹ {overall_avg_psf:.2f}")
        st.write(f"- Median Property Price (dataset): ₹ {overall_med_price:.2f} Lakhs")

        c1, c2 = st.columns([1,1])
        with c1:
            st.caption("Dataset Price per SqFt distribution")
            fig_d1 = px.histogram(df, x="Price_per_SqFt", nbins=50, title="Dataset Price per SqFt",
                                  color_discrete_sequence=[PALETTE[0]])
            fig_d1.update_layout(xaxis_title="Price per SqFt", yaxis_title="Count", height=360)
            st.plotly_chart(fig_d1, use_container_width=True)
        with c2:
            st.caption("Dataset BHK distribution")
            if "BHK" in df.columns:
                bhk_counts = df.groupby("BHK").size().reset_index(name="Count")
                bhk_counts["BHK"] = bhk_counts["BHK"].astype(str)
            else:
                bhk_counts = pd.DataFrame({"BHK": [], "Count": []})
            fig_d2 = px.bar(bhk_counts, x="BHK", y="Count", color="BHK",
                            color_discrete_sequence=PALETTE, title="Dataset BHK distribution")
            fig_d2.update_layout(xaxis_title="BHK", yaxis_title="Count", height=360, showlegend=False)
            st.plotly_chart(fig_d2, use_container_width=True)

st.markdown("---")
st.caption("Professional Dashboard — RealEstateInvestmentAdvisor")