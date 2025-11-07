
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Personal Loan Propensity • Universal Bank", layout="wide")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace('.', '').replace(' ', '_') for c in df.columns]
    return df

def prepare_xy(df: pd.DataFrame):
    drop_cols = [c for c in df.columns if c.lower() in ('id','zip','zipcode','zip_code','zip__code')]
    if 'Personal_Loan' not in df.columns and 'Personal Loan' in df.columns:
        df = df.rename(columns={'Personal Loan':'Personal_Loan'})
    X = df.drop(columns=drop_cols + ['Personal_Loan'])
    y = df['Personal_Loan']
    return X, y, drop_cols

def acceptance_rate_by(df: pd.DataFrame, col: str):
    grp = df.groupby(col)['Personal_Loan'].mean().sort_index()
    cnt = df.groupby(col)['Personal_Loan'].count().sort_index()
    return grp, cnt

def train_models(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "Gradient Boosted": GradientBoostingClassifier(n_estimators=200, random_state=random_state),
    }
    metrics = []
    rocs = {}
    cms = {}
    importances = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_tr_pred = mdl.predict(X_train)
        y_te_pred = mdl.predict(X_test)
        y_tr_proba = mdl.predict_proba(X_train)[:,1]
        y_te_proba = mdl.predict_proba(X_test)[:,1]
        row = {
            "Algorithm": name,
            "Train Accuracy": accuracy_score(y_train, y_tr_pred),
            "Test Accuracy": accuracy_score(y_test, y_te_pred),
            "Precision": precision_score(y_test, y_te_pred, zero_division=0),
            "Recall": recall_score(y_test, y_te_pred, zero_division=0),
            "F1": f1_score(y_test, y_te_pred, zero_division=0),
            "AUC": roc_auc_score(y_test, y_te_proba),
        }
        metrics.append(row)
        fpr, tpr, _ = roc_curve(y_test, y_te_proba)
        rocs[name] = (fpr, tpr, row["AUC"])
        cms[name] = {
            "train": confusion_matrix(y_train, y_tr_pred),
            "test": confusion_matrix(y_test, y_te_pred),
        }
        if hasattr(mdl, "feature_importances_"):
            importances[name] = mdl.feature_importances_
        else:
            importances[name] = np.zeros(X.shape[1])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_add = {}
    for name, base in models.items():
        cv_acc = cross_val_score(base, X, y, cv=cv, scoring='accuracy')
        cv_auc = cross_val_score(base, X, y, cv=cv, scoring='roc_auc')
        cv_add[name] = (cv_acc.mean(), cv_acc.std(), cv_auc.mean(), cv_auc.std())
    metrics_df = pd.DataFrame(metrics).set_index("Algorithm").round(4)
    metrics_df["CV5_Acc_Mean"] = [cv_add[n][0] for n in metrics_df.index]
    metrics_df["CV5_Acc_SD"] = [cv_add[n][1] for n in metrics_df.index]
    metrics_df["CV5_AUC_Mean"] = [cv_add[n][2] for n in metrics_df.index]
    metrics_df["CV5_AUC_SD"] = [cv_add[n][3] for n in metrics_df.index]
    return models, metrics_df, rocs, cms, importances, (X_train, X_test, y_train, y_test)

def render_confusion_matrix(cm, title, cmap):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm, cmap=cmap, interpolation='nearest')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Loan","Loan"]); ax.set_yticklabels(["No Loan","Loan"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center')
    st.pyplot(fig, use_container_width=True)

st.sidebar.title("Universal Bank — Loan Propensity")
st.sidebar.write("Upload a dataset or use the included sample to explore, model and predict.")

uploaded = st.sidebar.file_uploader("Upload CSV (same schema as sample)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("UniversalBank.csv")

df = clean_columns(df)

st.title("Personal Loan Propensity Dashboard")

tab1, tab2, tab3 = st.tabs([
    "Customer Insights (5 charts)",
    "Modeling: Train & Evaluate",
    "Score New Data & Download"
])

with tab1:
    st.subheader("Customer Insights")
    st.caption("Five complementary charts to guide marketing actions and targeting.")

    tmp = df.copy()
    tmp['IncomeDecile'] = pd.qcut(tmp['Income'], 10, labels=False, duplicates='drop')
    inc_rate = tmp.groupby('IncomeDecile')['Personal_Loan'].mean()
    inc_cnt = tmp.groupby('IncomeDecile')['Personal_Loan'].count()

    colA, colB = st.columns(2)
    with colA:
        fig, ax = plt.subplots()
        ax.bar(inc_rate.index.astype(str), inc_rate.values)
        ax.set_title("Acceptance Rate by Income Decile")
        ax.set_xlabel("Income Decile (0=lowest)"); ax.set_ylabel("Acceptance rate")
        st.pyplot(fig, use_container_width=True)
        st.markdown("- **Action**: Prioritize higher-income deciles with personalized cross-sell.")

    with colB:
        fig, ax = plt.subplots()
        ax.bar(inc_cnt.index.astype(str), inc_cnt.values)
        ax.set_title("Customer Volume by Income Decile")
        ax.set_xlabel("Income Decile"); ax.set_ylabel("Number of customers")
        st.pyplot(fig, use_container_width=True)
        st.markdown("- **Action**: Balance high-rate segments with high-volume ones for total conversions.")

    edu_fam = df.pivot_table(index='Education', columns='Family', values='Personal_Loan', aggfunc='mean')
    fig, ax = plt.subplots()
    im = ax.imshow(edu_fam.values, aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(edu_fam.columns))); ax.set_xticklabels(edu_fam.columns)
    ax.set_yticks(range(len(edu_fam.index))); ax.set_yticklabels(edu_fam.index)
    ax.set_xlabel("Family Size"); ax.set_ylabel("Education (1 UG, 2 Grad, 3 Adv)")
    ax.set_title("Acceptance Rate: Education × Family (Heatmap)")
    st.pyplot(fig, use_container_width=True)
    st.markdown("- **Action**: Tailor messaging for **Education=3** with family size 3–4 (often higher rates).")

    colors = df['Personal_Loan'].map({0:'gray',1:'red'})
    fig, ax = plt.subplots()
    ax.scatter(df['Income'], df['CCAvg'], c=colors, alpha=0.5, s=10)
    ax.set_title("Income vs CCAvg (red = accepted loan)")
    ax.set_xlabel("Income ($000)"); ax.set_ylabel("CCAvg ($000)")
    st.pyplot(fig, use_container_width=True)
    st.markdown("- **Action**: High income & mid–high CCAvg show dense positive pockets → premium offers.")

    roll = df.sort_values('Age').groupby('Age')['Personal_Loan'].mean().rolling(5, min_periods=1).mean()
    fig, ax = plt.subplots()
    ax.plot(roll.index, roll.values)
    ax.set_title("Smoothed Acceptance Rate by Age")
    ax.set_xlabel("Age"); ax.set_ylabel("Acceptance rate (rolling mean)")
    st.pyplot(fig, use_container_width=True)
    st.markdown("- **Action**: Identify age bands where propensity peaks for age-specific creatives.")

    num = df.select_dtypes(include=[np.number]).drop(columns=[c for c in ['Personal_Loan'] if c in df.columns])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(corr.values, aspect='auto')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Matrix (numeric features)")
    st.pyplot(fig, use_container_width=True)
    st.markdown("- **Action**: Watch multicollinearity (e.g., **Income** with **CCAvg**).")

with tab2:
    st.subheader("Train & Evaluate Models")
    st.write("Click **Train Models** to run Decision Tree, Random Forest, and Gradient Boosted Tree with 5-fold CV.")
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            X, y, dropped = prepare_xy(df)
            models, metrics_df, rocs, cms, importances, splits = train_models(X, y)
        st.success("Done!")
        st.write("### Performance Metrics")
        st.dataframe(metrics_df)

        st.write("### ROC Curves (combined)")
        fig, ax = plt.subplots(figsize=(6,5))
        colors = {"Decision Tree":"blue","Random Forest":"green","Gradient Boosted":"red"}
        for name, (fpr,tpr,auc_val) in rocs.items():
            ax.plot(fpr, tpr, label=f\"{name} (AUC={auc_val:.3f})\", color=colors[name])
        ax.plot([0,1],[0,1],'--', linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curves")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        st.write("### Confusion Matrices")
        for name, mats in cms.items():
            c1,c2 = st.columns(2)
            with c1:
                render_confusion_matrix(mats['train'], f"{name} — TRAIN", plt.cm.Blues)
            with c2:
                render_confusion_matrix(mats['test'], f"{name} — TEST", plt.cm.Oranges)

        st.write("### Feature Importances")
        feat_names = list(X.columns)
        for name, imps in importances.items():
            order = np.argsort(imps)[::-1]
            fig, ax = plt.subplots(figsize=(8,3.5))
            ax.bar(range(len(imps)), imps[order])
            ax.set_xticks(range(len(imps))); ax.set_xticklabels(np.array(feat_names)[order], rotation=90)
            ax.set_ylabel("Importance"); ax.set_title(f"Feature Importances — {name}")
            st.pyplot(fig, use_container_width=True)

        st.session_state["models"] = models
        st.session_state["train_columns"] = list(X.columns)

with tab3:
    st.subheader("Upload & Predict")
    st.write("Upload a **new CSV** with the same schema to get predictions. You can choose which trained model to use.")
    models = st.session_state.get("models", None)
    col_sel1, _ = st.columns([2,1])
    with col_sel1:
        if models:
            model_name = st.selectbox("Select trained model", list(models.keys()), index=1)
        else:
            model_name = None
            st.info("Train models in the previous tab to enable model selection.")

    new_file = st.file_uploader("Upload new CSV to score", type=["csv"], key="scorecsv")
    if new_file is not None:
        new_df_raw = pd.read_csv(new_file)
        new_df = clean_columns(new_df_raw)
        st.write("Preview:", new_df.head())

        if models and model_name:
            mdl = models[model_name]
            X_new, _, dropped = prepare_xy(new_df.assign(Personal_Loan=0))
            train_cols = st.session_state.get("train_columns", list(X_new.columns))
            for c in train_cols:
                if c not in X_new.columns:
                    X_new[c] = 0
            X_new = X_new[train_cols]
            preds = mdl.predict(X_new)
            proba = mdl.predict_proba(X_new)[:,1]
            out = new_df.copy()
            out["Personal_Loan_Pred"] = preds
            out["Personal_Loan_Prob"] = np.round(proba, 4)
            st.write("Scored Sample:", out.head())
            csv = out.to_csv(index=False).encode()
            st.download_button("Download Scored CSV", data=csv, file_name="scored_personal_loan.csv", mime="text/csv")
        else:
            st.warning("Please train models first in the previous tab.")

st.caption("Tip: Flat files only (no folders) for smooth GitHub → Streamlit Cloud deployment.")
