from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import threading
import time

app = Flask(__name__)

# Define parameter ranges
ranges = {
    'casting_temperature': (680, 750),  # °C
    'cooling_temp': (20, 35),           # °C
    'casting_speed': (6, 12),           # m/min
    'entry_temp': (400, 500),           # °C
    'emulsion_temp': (40, 60),          # °C
    'emulsion_pressure': (1, 3),        # bars
    'emulsion_concentration': (2, 5),   # %
    'quench_pressure': (1, 2)           # bars
}

# Generate random data within the specified ranges
def generate_random_data(num_samples):
    data = {param: np.random.uniform(low, high, num_samples) for param, (low, high) in ranges.items()}
    df = pd.DataFrame(data)
    return df

# Simulate quality metric (for demonstration)
def simulate_quality(df):
    # Simple function to simulate the quality metric based on input parameters
    quality = (
        0.3 * df['casting_temperature'] +
        0.2 * df['cooling_temp'] +
        0.1 * df['casting_speed'] +
        0.15 * df['entry_temp'] +
        0.05 * df['emulsion_temp'] +
        0.1 * df['emulsion_pressure'] +
        0.05 * df['emulsion_concentration'] +
        0.05 * df['quench_pressure']
    )
    return quality

# Generate training data
num_samples = 1000
df = generate_random_data(num_samples)
df['quality'] = simulate_quality(df)

# Train a regression model
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model MSE: {mse}")

# Real-time data storage
real_time_params = {}

# Function to monitor and adjust parameters
def monitor_and_adjust():
    global real_time_params
    while True:
        if real_time_params:
            # Convert current_params dict to DataFrame for prediction
            current_df = pd.DataFrame([real_time_params])
            predicted_quality = model.predict(current_df)[0]
            print(f"Predicted Quality: {predicted_quality:.2f}")

            # Check if any parameter is out of range and adjust
            adjustments_needed = False
            for param, value in real_time_params.items():
                low, high = ranges[param]
                if value < low or value > high:
                    adjustments_needed = True
                    print(f"{param} out of range! Adjusting...")
                    # Adjust the parameter to be within the range
                    real_time_params[param] = np.clip(value, low, high)
            
            if adjustments_needed:
                # Recalculate quality with adjusted parameters
                adjusted_df = pd.DataFrame([real_time_params])
                adjusted_quality = model.predict(adjusted_df)[0]
                print(f"Adjusted Quality: {adjusted_quality:.2f}")

            # Simulate real-time parameter changes
            real_time_params = {k: v + np.random.uniform(-1, 1) for k, v in real_time_params.items()}

        time.sleep(1)

# Start monitoring in a separate thread
threading.Thread(target=monitor_and_adjust, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    global real_time_params
    return jsonify(real_time_params)

@app.route('/set_data', methods=['POST'])
def set_data():
    global real_time_params
    real_time_params = request.json
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(debug=True)