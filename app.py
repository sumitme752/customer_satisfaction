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
                    "Departure/Arrival time convenient": int(request.form["departure_arrival_convenience"]),
                    "Inflight wifi service": int(request.form["inflight_wifi_service"]),
                    "Cleanliness": int(request.form["cleanliness"]),
                    "Baggage handling": int(request.form["baggage_handling"]),
                    "Flight Distance": int(request.form["flight_distance"]),
                    "Leg room service": int(request.form["leg_room_service"]),
                    "Departure Delay in Minutes": int(request.form["departure_delay_in_minutes"]),
                    "Arrival Delay in Minutes": float(request.form["arrival_delay_in_minutes"]),
                    "Ease of Online booking": int(request.form["ease_of_online_booking"]),
                    "Checkin service": int(request.form["checkin_service"]),
                    "Age": int(request.form["age"]),
                    "On-board service": int(request.form["onboard_service"]),
                    "Online boarding": int(request.form["online_boarding"]),
                    "Gate location": int(request.form["gate_location"]),
                    "Food and drink": int(request.form["food_and_drink"]),
                    "Customer Type": request.form["customer_type"],
                    "Type of Travel": request.form["type_of_travel"],
                    "Inflight service": int(request.form["inflight_service"]),
                    "Seat comfort": int(request.form["seat_comfort"]),
                    "Inflight entertainment": int(request.form["inflight_entertainment"]),
                    "Class": request.form["class"],
                    }


            # Store the feedback (optional, or use it for logging/debugging)
            feedback_data.append(data)
            # print(feedback_data)


            # Create a DataFrame for the current input
            # Create DataFrame
            predict_df = pd.DataFrame(feedback_data)

            # Make prediction
            prediction = model.predict(predict_df)  # Get the first prediction
            print(prediction)

            # To convert to a single value or list
            single_prediction = prediction.tolist()[-1]

            # Map prediction result to a meaningful output
            result = 'Satisfied' if single_prediction == 1 else 'Neutral or Dissatisfied'

            return render_template("result.html", result=result)

        except Exception as e:
            return f"An error occurred: {str(e)}", 400

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
