from flask import Flask, render_template, request, redirect, url_for, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the model once during initialization
MODEL_PATH = './model/my_model.pkl'
model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# In-memory storage for demonstration (you can replace this with a DB)
feedback_data = []

@app.route("/", methods=["GET", "POST"])
def feedback_form():
    if request.method == "POST":
        try:
            data = {
                "customer_type": request.form["customer_type"],
                "age": int(request.form["age"]),
                "type_of_travel": request.form["type_of_travel"],
                "class": request.form["class"],
                "flight_distance": int(request.form["flight_distance"]),
                "inflight_wifi_service": int(request.form["inflight_wifi_service"]),
                "departure_arrival_convenience": int(request.form["departure_arrival_convenience"]),
                "ease_of_online_booking": int(request.form["ease_of_online_booking"]),
                "gate_location": int(request.form["gate_location"]),
                "food_and_drink": int(request.form["food_and_drink"]),
                "online_boarding": int(request.form["online_boarding"]),
                "seat_comfort": int(request.form["seat_comfort"]),
                "inflight_entertainment": int(request.form["inflight_entertainment"]),
                "onboard_service": int(request.form["onboard_service"]),
                "leg_room_service": int(request.form["leg_room_service"]),
                "baggage_handling": int(request.form["baggage_handling"]),
                "checkin_service": int(request.form["checkin_service"]),
                "inflight_service": int(request.form["inflight_service"]),
                "cleanliness": int(request.form["cleanliness"]),
                "departure_delay_in_minutes": int(request.form["departure_delay_in_minutes"]),
                "arrival_delay_in_minutes": float(request.form["arrival_delay_in_minutes"]),
            }

            # Store the feedback (optional, or use it for logging/debugging)
            feedback_data.append(data)
            print(feedback_data)

            # Create a DataFrame for the current input
            predict_df = pd.DataFrame([data])

            # Make prediction
            prediction = model.predict(predict_df)[0]  # Get the first prediction

            # Map prediction result to a meaningful output
            result = 'Satisfied' if prediction == 1 else 'Neutral or Dissatisfied'

            return render_template("result.html", result=result)

        except Exception as e:
            return f"An error occurred: {str(e)}", 400

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
