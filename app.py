from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the data
data = pd.read_csv("heart.csv")

# Selecting all features (except the target)
X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = data['target']

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Read HTML file
with open("index.html", "r", encoding="utf-8") as file:
    HTML_TEMPLATE = file.read()

# Home route
@app.route("/", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        return render_template_string(HTML_TEMPLATE, page="details")
    return render_template_string(HTML_TEMPLATE, page="register")

@app.route("/details", methods=["POST"])
def details():
    try:
        # Collect form data from the user
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        cp = int(request.form["cp"])  # Chest pain type (0-3)
        trestbps = int(request.form["trestbps"])  # Blood pressure
        chol = int(request.form["chol"])  # Cholesterol level
        fbs = int(request.form["fbs"])  # Fasting blood sugar > 120 mg/dl
        restecg = int(request.form["restecg"])  # Resting electrocardiographic results
        thalach = int(request.form["thalach"])  # Maximum heart rate achieved
        exang = int(request.form["exang"])  # Exercise induced angina (1 = yes; 0 = no)
        oldpeak = float(request.form["oldpeak"])  # ST depression induced by exercise relative to rest
        slope = int(request.form["slope"])  # Slope of peak exercise ST segment
        ca = int(request.form["ca"])  # Number of major vessels colored by fluoroscopy
        thal = int(request.form["thal"])  # Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)

        # Prepare user input for prediction
        user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Scale the user input to match the training data
        user_data_scaled = scaler.transform(user_data)

        # Make the prediction
        prediction = model.predict(user_data_scaled)[0]

        # Generate the result message
        result = "Heart Disease Detected! Consult a doctor." if prediction == 1 else "No Heart Disease Detected!"

        # Render the result page
        return render_template_string(HTML_TEMPLATE, page="result", result=result)

    except KeyError as e:
        return f"Missing form data: {str(e)}", 400

if __name__ == "_main_":
    app.run(debug=True)