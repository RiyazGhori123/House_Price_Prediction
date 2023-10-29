import pandas as pd
from flask import Flask, render_template, request
import pickle

app=Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data from the POST request
        location = request.form['location']
        bhk = float(request.form['bhk'])
        bath = float(request.form['bath'])
        total_sqft = float(request.form['total_sqft'])

        # You can calculate the predicted price based on your criteria here
        # For example, you can define your pricing logic and calculate the price

        # Here's a simple example of price calculation based on your inputs
        # This is a placeholder and should be replaced with your actual pricing logic
        predicted_price = bhk * 1000 + bath * 200 + total_sqft * 3000

        # Return the calculated price as a response
        return str(predicted_price)  # Convert the calculated price to a string

if __name__ == "__main__":
    app.run(debug=True, port=5001)