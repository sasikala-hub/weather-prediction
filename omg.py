import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit App Title
st.title("Weather Data - Pressure Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Required Columns
    features = [
        'Temperature (C)', 'Wind Speed (km/h)', 'Apparent Temperature (C)',
        'Humidity', 'Wind Bearing (degrees)', 'Visibility (km)'
    ]
    target = 'Pressure (millibars)'

    # Check for missing columns
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {missing_cols}")
    else:
        st.write("### Data Preview")
        st.dataframe(df.head())

        # Show correlation heatmap
        st.write("### Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[features + [target]].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Splitting data
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display Model Performance
        st.write("### Model Performance")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

        # Prediction on User Input
        st.write("### Predict Pressure for New Data")
        input_data = {}
        for feature in features:
            input_data[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))

        if st.button("Predict Pressure"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.success(f"Predicted Pressure: {prediction[0]:.2f} millibars")
