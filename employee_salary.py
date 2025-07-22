import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- Page Configuration ---
st.set_page_config(page_title="Employer Salary Predictor", page_icon="💼", layout="centered")

# --- Load & Generate Synthetic Data ---
@st.cache_data
def load_data():
    np.random.seed(42)
    job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'HR Executive', 'Marketing Specialist']
    education_levels = ['High School', 'Diploma', 'Bachelors', 'Masters', 'PhD']
    departments = ['IT', 'HR', 'Marketing', 'Finance', 'Operations']
    locations = ['New York', 'San Francisco', 'Chicago', 'Austin', 'India', 'USA', 'Australia', 'Saudi Arabia', 'China', 'Sri Lanka', 'London', 'Berlin', 'Tokyo', 'Sydney']
    work_types = ['Remote', 'In-office', 'Hybrid', 'On-site']
    job_types = ['Full-time', 'Part-time', 'Internship', 'Freelance']

    data = {
        'Job Title': np.random.choice(job_titles, 1000),
        'Education Level': np.random.choice(education_levels, 1000),
        'Department': np.random.choice(departments, 1000),
        'Location': np.random.choice(locations, 1000),
        'Work Type': np.random.choice(work_types, 1000),
        'Job Type': np.random.choice(job_types, 1000),
        'Years of Experience': np.random.randint(0, 16, 1000),
    }

    df = pd.DataFrame(data)
    base_salary = 35000
    df['Salary'] = base_salary + \
                   df['Years of Experience'] * 1500 + \
                   df['Job Title'].apply(lambda x: job_titles.index(x) * 6000) + \
                   df['Job Type'].apply(lambda x: job_types.index(x) * 3000) + \
                   np.random.randint(0, 15000, 1000)
    return df

df = load_data()

# --- Encode Categorical Columns ---
def encode_features(df, encoders=None):
    if encoders is None:
        encoders = {}
        for col in ['Job Title', 'Education Level', 'Department', 'Location', 'Work Type', 'Job Type']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in encoders:
            df[col] = encoders[col].transform(df[col])
    return df, encoders

df_encoded, label_encoders = encode_features(df.copy())

# --- Train Model ---
X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

r2 = r2_score(y_test, model.predict(X_test))

# --- App Title ---
st.title("Employee Salary Prediction")

st.markdown("Estimate salary using employee background and job profile powered by **Random Forest** regression.")

# --- Sidebar Info ---
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown(f"""
    - **Model:** Random Forest Regressor  
    - **Records:** 1000 synthetic  
    - **R² Score:** {r2:.2f}  
    """)
    st.info("Model trained live in app!")

# --- Input Form ---
with st.form("predict_form"):
    st.subheader("Employee Profile Details:")

    col1, col2 = st.columns(2)
    with col1:
        job = st.selectbox("🧑‍💻 Job Title", label_encoders['Job Title'].classes_)
        edu = st.selectbox("🎓 Education Level", label_encoders['Education Level'].classes_)
        dept = st.selectbox("🏢 Department", label_encoders['Department'].classes_)

    with col2:
        loc = st.selectbox("🌍 Location", label_encoders['Location'].classes_)
        work = st.selectbox("🏠 Work Type", label_encoders['Work Type'].classes_)
        job_type = st.selectbox("🕒 Job Type", label_encoders['Job Type'].classes_)

    # Experience Type
    exp_type = st.selectbox("🧑‍🏫 Experience Type", ['Intern', 'Fresher', 'Experienced'])

    if exp_type == "Intern":
        experience = 0
    elif exp_type == "Fresher":
        experience = 1
    else:
        experience = st.slider("🔢 Years of Experience", 1, 30, 3)

    submitted = st.form_submit_button("💰 Predict Salary")

# --- Make Prediction ---
if submitted:
    input_data = pd.DataFrame({
        'Job Title': [job],
        'Education Level': [edu],
        'Department': [dept],
        'Location': [loc],
        'Work Type': [work],
        'Job Type': [job_type],
        'Years of Experience': [experience]
    })

    for col in input_data.columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)[0]
    st.success(f"🤑 Estimated Salary: **${prediction:,.2f}**")

# --- Show Distribution ---
with st.expander("📈 Salary Distribution in Dataset"):
    fig, ax = plt.subplots()
    sns.histplot(df["Salary"], kde=True, bins=30, ax=ax, color='skyblue')
    if submitted:
        ax.axvline(prediction, color='red', linestyle='--', label="Predicted Salary")
        ax.legend()
    ax.set_title("Overall Salary Distribution")
    st.pyplot(fig)

# --- Footer ---
st.markdown("""
<hr style='margin-top:40px; margin-bottom:10px;'>
<div style='text-align:center; color:gray;'>Created by <b>Mukesh – AI/ML Intern, Edunet Foundation</b></div>
""", unsafe_allow_html=True)
