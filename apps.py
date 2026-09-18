import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load the trained model
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("Survival.pkl")

    # --- Compatibility patch ---
    # Models pickled with older scikit-learn versions can be missing
    # attributes expected by a newer scikit-learn at predict time
    # (e.g. 'multi_class' on LogisticRegression). Patch them back in
    # so predict()/predict_proba() don't raise AttributeError.
    try:
        clf = model.named_steps["classifier"]
    except AttributeError:
        clf = model  # in case a bare estimator was saved, not a pipeline

    if not hasattr(clf, "multi_class"):
        clf.multi_class = "auto"
    if not hasattr(clf, "n_jobs"):
        clf.n_jobs = None
    if not hasattr(clf, "l1_ratio"):
        clf.l1_ratio = None

    return model

model = load_model()

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter a passenger's details below to predict whether they would have "
    "survived the Titanic disaster."
)

# ----------------------------
# Input widgets
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2)
    sex = st.selectbox("Sex", options=["Female", "Male"])
    age = st.slider("Age", min_value=0, max_value=80, value=30)
    fare = st.number_input("Fare Paid ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    embarked = st.selectbox("Port of Embarkation", options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# ----------------------------
# Encode inputs to match training features
# Model expects columns: ['Pclass', 'Age', 'SibSp', 'Fare', 'Parch', 'Gender', 'C', 'Q']
# Gender: Female = 1, Male = 0
# Embarked: one-hot for C and Q; Southampton (S) is the baseline (both 0)
# ----------------------------
gender = 1 if sex == "Female" else 0
embarked_c = 1 if embarked.startswith("Cherbourg") else 0
embarked_q = 1 if embarked.startswith("Queenstown") else 0

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Fare": fare,
    "Parch": parch,
    "Gender": gender,
    "C": embarked_c,
    "Q": embarked_q,
}])

st.subheader("Input Summary")
st.dataframe(input_df, use_container_width=True)

# ----------------------------
# Predict
# ----------------------------
if st.button("Predict Survival", type="primary"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    survival_prob = probability[1] * 100
    death_prob = probability[0] * 100

    if prediction == 1:
        st.success(f"✅ This passenger would likely **SURVIVE** ({survival_prob:.1f}% probability)")
    else:
        st.error(f"❌ This passenger would likely **NOT SURVIVE** ({death_prob:.1f}% probability)")

    st.write("### Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Outcome": ["Did Not Survive", "Survived"],
        "Probability (%)": [death_prob, survival_prob],
    })
    st.bar_chart(prob_df.set_index("Outcome"))

st.markdown("---")
st.caption(
    "Model: Logistic Regression trained on the Titanic dataset. "
    "Place `Survival.pkl` in the same directory as this script to run it "
    "with: `streamlit run survived.py`"
)