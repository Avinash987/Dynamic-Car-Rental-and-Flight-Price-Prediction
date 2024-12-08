import streamlit as st
import pandas as pd
import pickle

# Load models
def load_models():
    with open("flight_model.pkl", "rb") as f:
        flight_model = pickle.load(f)
    with open("car_model.pkl", "rb") as f:
        car_model = pickle.load(f)
    return flight_model, car_model

flight_model, car_model = load_models()

# Page layout
st.title("Price Prediction App")
st.sidebar.header("Choose Prediction Type")
prediction_type = st.sidebar.selectbox("Select Prediction Model", 
                                       ["Flight Price Prediction", "Car Rental Price Prediction"])

# Function to align input features
def align_features(input_df, model_features):
    # Add missing features as zero
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    # Drop extra columns not seen during training
    input_df = input_df[model_features]
    return input_df

# Flight Price Prediction
if prediction_type == "Flight Price Prediction":
    st.header("Flight Price Prediction")
    
    # User inputs
    starting_airport = st.text_input("Starting Airport (e.g., LAX)")
    destination_airport = st.text_input("Destination Airport (e.g., JFK)")
    travel_duration = st.number_input("Travel Duration (in hours)", min_value=0.0, step=0.1)
    is_nonstop = st.selectbox("Non-Stop Flight?", ["Yes", "No"])
    is_refundable = st.selectbox("Refundable Ticket?", ["Yes", "No"])
    airline_name = st.text_input("Airline Name (e.g., Delta)")

    if st.button("Predict Flight Price"):
        # Prepare raw input
        input_data = pd.DataFrame([{
            "startingAirport": starting_airport,
            "destinationAirport": destination_airport,
            "travelDuration": travel_duration,
            "isNonStop": 1 if is_nonstop == "Yes" else 0,
            "isRefundable": 1 if is_refundable == "Yes" else 0,
            "segmentsAirlineName": airline_name
        }])

        # Align input with model features
        try:
            model_features = flight_model.feature_names_in_  # Extract feature names from the model
            input_data = align_features(input_data, model_features)
            flight_price = flight_model.predict(input_data)[0]
            st.success(f"Predicted Flight Price: ${flight_price:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")



# Car Rental Price Prediction
if prediction_type == "Car Rental Price Prediction":
    st.header("Car Rental Price Prediction")
    
    # User inputs
    fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Electric", "Hybrid", "Diesel"])
    rating = st.number_input("Vehicle Rating (1-5)", min_value=1.0, max_value=5.0, step=0.1)
    renter_trips_taken = st.number_input("Renter Trips Taken", min_value=0, step=1)
    vehicle_year = st.number_input("Vehicle Year", min_value=2000, max_value=2024, step=1)
    review_count = st.number_input("Review Count", min_value=0, step=1)

    if st.button("Predict Rental Price"):
        # Prepare raw input
        fuel_type_mapping = {"Gasoline": 0, "Electric": 1, "Hybrid": 2, "Diesel": 3}
        encoded_fuel_type = fuel_type_mapping[fuel_type]

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "fuelType": encoded_fuel_type,
            "rating": rating,
            "renterTripsTaken": renter_trips_taken,
            "vehicle.year": vehicle_year,
            "reviewCount": review_count
        }])

        try:
            # Attempt to get expected features from the model
            try:
                expected_features = car_model.get_booster().feature_names
            except AttributeError:
                expected_features = None
            
            # Fallback: Define expected features manually if not available
            if not expected_features:
                expected_features = [
                    "fuelType", "rating", "renterTripsTaken", "vehicle.year",
                    "reviewCount", "location.state", "vehicle.type",
                    "weekday", "season", "holiday"
                ]

            # Add missing features with default values
            for feature in expected_features:
                if feature not in input_data.columns:
                    input_data[feature] = 0  # Default value for missing features
            
            # Ensure columns are in the correct order
            input_data = input_data[expected_features]

            # Debugging: Print the input data
            st.write("Aligned Input DataFrame:")
            st.write(input_data)

            # Predict price
            car_price = car_model.predict(input_data)[0]
            st.success(f"Predicted Car Rental Price: ${car_price:.2f} per day")
        except Exception as e:
            st.error(f"Error: {e}")