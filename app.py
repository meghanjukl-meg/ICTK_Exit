import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
model = joblib.load('bengaluru_model.pkl')
locations = joblib.load('locations.pkl')

# encodings
le = joblib.load('location_encoder.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('cleaned_data.csv')
st.title("Bengaluru House Price Predictor")

# --- Sidebar: User Inputs ---
st.sidebar.header("Property Details")
loc = st.sidebar.selectbox("Select Location", locations)
sqft = st.sidebar.number_input(
    "Total Square Feet", min_value=1.0, value=1000.0)
bhk = st.sidebar.slider("BHK", 1, 10, 2)
bath = st.sidebar.slider("Bathrooms", 1, 10, 2)
balcony = st.sidebar.selectbox("Balcony Count", [0, 1, 2, 3])
# available = st.sidebar.selectbox("Availability",[0,1])

if st.sidebar.button("Predict Price"):

    if sqft <= 0:
        st.error("Squarefeet must be greater than zero!")

    else:
        loc_encoded = le.transform([loc])[0]
        scaled_values = scaler.transform([[sqft, bath]])
        sqft_scaled = scaled_values[0][0]
        bath_scaled = scaled_values[0][1]

        input_data = pd.DataFrame([[loc_encoded, sqft_scaled, bath_scaled, balcony, bhk]],
                                  columns=['location', 'total_sqft', 'bath', 'balcony', 'bhk'])

        # If you used Label Encoding/Scaling, apply them here first
        prediction = model.predict(input_data)[0]
        st.success(f"### Estimated Price: ₹ {round(prediction, 2)} Lakhs")

# --- Visualization: Top 5 Expensive Locations ---
st.subheader(f"Top 5 Expensive Locations for {bhk} BHK")

chart_data = df[df['bhk'] == bhk].groupby(
    'location')['price'].mean().nlargest(5).reset_index()
fig = px.bar(chart_data, x='location', y='price', color='price',
             labels={'price': 'Avg Price (Lakhs)'})
st.plotly_chart(fig)

st.info("Input validation active: Square footage must be positive.")
