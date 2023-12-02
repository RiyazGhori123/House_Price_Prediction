# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import pickle

# app = Flask(__name__)
# data = pd.read_csv('C:/Users/Dell/Desktop/projects/4th year/Capstone/Capstone Project/House_Price_Prediction/PycharmProjects/Cleaned_data.csv')

# # Load the pickled model in binary mode
# with open('C:/Users/Dell/Desktop/projects/4th year/Capstone/Capstone Project/House_Price_Prediction/PycharmProjects/RidgeModel.pkl', 'rb') as file:
#     pipe = pickle.load(file)

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/index', methods=['GET', 'POST'])
# def index():
#     locations = sorted(data['location'].unique())
#     prediction = None  # Initialize prediction

#     if request.method == 'POST':
#         location = request.form.get('location')
#         bhk = request.form.get('bhk')
#         bath = request.form.get('bath')
#         sqft = request.form.get('totalsqft')

#         # Validate input fields
#         if not location or not bhk or not bath or not sqft:
#             return render_template('index.html', locations=locations, prediction="Please fill in all fields")

#         try:
#             # Try to convert input values to float
#             location = float(location)
#             bhk = float(bhk)
#             bath = float(bath)
#             sqft = float(sqft)

#             input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
#             prediction = pipe.predict(input_data)[0]  # Get the prediction
#         except ValueError:
#             # Handle invalid input values (e.g., non-numeric input)
#             return render_template('index.html', locations=locations, prediction="Invalid input values")

#     return render_template('index.html', locations=locations, prediction=prediction)

# @app.route('/predict', methods=['POST'])
# def predict():
#     location = request.form.get('location')
#     bhk = float(request.form.get('bhk'))
#     bath = float(request.form.get('bath'))
#     total_sqft = float(request.form.get('totalsqft'))

#     # Calculate the predicted price using the custom logic
#     predicted_price = bhk * 100 + bath * 200 + total_sqft * 300

#     # Format the prediction to 3 decimal places and return as JSON
#     return f'Predicted Price: Rs.{predicted_price:.3f}'


# if __name__ == "__main__":
#     app.run(debug=True, port=5002)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('./PycharmProjects/Cleaned_data.csv')

# Load the pickled model in binary mode
with open('./PycharmProjects/RidgeModel.pkl', 'rb') as file:
    pipe = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    locations = sorted(data['location'].unique())
    prediction = None  # Initialize prediction

    if request.method == 'POST':
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath = request.form.get('bath')
        sqft = request.form.get('totalsqft')

        # Validate input fields
        if not location or not bhk or not bath or not sqft:
            return render_template('index.html', locations=locations, prediction="Please fill in all fields")

        try:
            # Try to convert input values to float
            location = float(location)
            bhk = float(bhk)
            bath = float(bath)
            sqft = float(sqft)

            input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
            prediction = pipe.predict(input_data)[0]  # Get the prediction
        except ValueError:
            # Handle invalid input values (e.g., non-numeric input)
            return render_template('index.html', locations=locations, prediction="Invalid input values")

    return render_template('index.html', locations=locations, prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    total_sqft = float(request.form.get('totalsqft'))

    # Use the pre-trained Ridge model to make predictions
    input_data = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    predicted_price = pipe.predict(input_data)[0]

    # Format the prediction to 3 decimal places and return as JSON
    return jsonify({'predicted_price': f'Rs. {predicted_price:.3f}'})

if __name__ == "__main__":
    app.run(debug=True, port=5002)
