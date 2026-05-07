import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------------------------
# 1. THE DATA
# ------------------------------------------------
data = {
    'Square_Feet': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Price': [150000, 220000, 250000, 320000, 350000] 
}
df = pd.DataFrame(data)

# ------------------------------------------------
# 2. THE MACHINE LEARNING MODEL
# ------------------------------------------------
X = df[['Square_Feet', 'Bedrooms']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# ------------------------------------------------
# 3. THE COLORFUL USER INTERFACE
# ------------------------------------------------
# Use :color[text] or :rainbow[text] to add instant color!
st.title("🏡 :violet[Simple] :rainbow[House Price Predictor]")

# st.info creates a nice blue box, st.warning is yellow, st.error is red
st.info("💡 **Tip:** Adjust the sliders below to see the estimated price change in real-time.")

st.markdown("### :blue[Step 1:] Tell us about the house")
sqft = st.slider("Select Square Footage", min_value=500, max_value=50000, value=1500, step=100)
beds = st.slider("Select Number of Bedrooms", min_value=1, max_value=10, value=3)

st.divider()

st.markdown("### :orange[Step 2:] Get your estimate")

# type="primary" makes the button adopt your app's main theme color (usually red or blue) instead of gray
if st.button("Predict Price", type="primary"):
    
    predicted_price = model.predict([[sqft, beds]])
    final_price = predicted_price[0]
    
    # st.success creates a nice green box
    st.success(f"### Estimated Price: :green[**${final_price:,.2f}**]")
    
    st.balloons()