import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# 🎨 CUSTOM CSS (COLOR DESIGN)
# ----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #e3f2fd, #fce4ec);
}

/* Title */
h1 {
    color: #6a1b9a;
    text-align: center;
}

/* Section headers */
h3 {
    color: #1565c0;
}

/* Buttons */
.stButton>button {
    background-color: #ff4081;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

/* Button hover */
.stButton>button:hover {
    background-color: #c2185b;
    color: white;
}

/* Result box */
.result-box {
    background-color:#e3f2fd;
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:24px;
    color:#0d47a1;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# 🏡 TITLE
# ----------------------------
st.title("🏡 :rainbow[ House Price Predictor]")

# ----------------------------
# 📌 SIDEBAR
# ----------------------------
st.sidebar.markdown("## 🎨 Menu")
st.sidebar.success("Welcome!")
st.sidebar.info("Edit data → Train → Predict")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📞 Contact")
st.sidebar.write("📍 LA,American")
st.sidebar.write("📞 +91 98765 43210")
st.sidebar.write("📧 support@housepredictor.com")

# ----------------------------
# 1. DATA INPUT
# ----------------------------
st.markdown("### :blue[Step 1: Enter your training data]")

initial_data = pd.DataFrame(
    columns=['Square_Feet', 'Bedrooms', 'Price', 'Address'],
    data=[
        [1000, 2, 150000, 'Downtown'],
        [1500, 3, 220000, 'Washington'],
        [2000, 3, 250000, 'LA'],
        [2500, 4, 320000, 'Whitehouse'],
        [3000, 4, 350000, 'Las Vegas']
    ]
)

df = st.data_editor(initial_data, num_rows="dynamic", use_container_width=True)

st.divider()

# ----------------------------
# MODEL TRAINING
# ----------------------------
if len(df) < 2:
    st.warning("⚠️ Please add at least 2 rows of data.")
else:
    # Encode location
    le = LabelEncoder()
    df['Address_Encoded'] = le.fit_transform(df['Address'])

    X = df[['Square_Feet', 'Bedrooms', 'Address_Encoded']]
    y = df['Price']

    model = LinearRegression()
    model.fit(X, y)

    st.success("✅ Model trained successfully!")

    # ----------------------------
    # USER INPUT
    # ----------------------------
    st.markdown("### :orange[Step 2: Predict Price]")

    sqft = st.slider("🏠 Square Feet", 500, 5000, 1500)
    beds = st.slider("🛏 Bedrooms", 1, 10, 3)

    location = st.selectbox("📍 Select Location", df['Address'].unique())
    location_encoded = le.transform([location])[0]

    # ----------------------------
    # PREDICTION
    # ----------------------------
    if st.button("🚀 Predict Price"):
        predicted_price = model.predict([[sqft, beds, location_encoded]])
        final_price = predicted_price[0]

        st.markdown(f"""
        <div class="result-box">
        💰 Estimated Price: ₹{final_price:,.2f}
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

# ----------------------------
# 📞 FOOTER CONTACT
# ----------------------------
st.divider()
st.markdown("## 📞 Contact Us")

st.markdown("""
📍 **Office Address:**  
123, Southwest american

📞 **Phone:** +91 98765 43210  

📧 **Email:** support@housepredictor.com  
""")