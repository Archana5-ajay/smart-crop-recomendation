from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load ML model and encoder
model = joblib.load('crop_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
df = pd.read_csv(r"C:\Users\archa\OneDrive\Desktop\ASSIGNMENTS\predictive analytics\Crop_recommendation.csv")

# Ensure static folder exists for charts
if not os.path.exists('static'):
    os.makedirs('static')

# ✅ Function to Generate Charts
def generate_charts():
    charts = [
        ("Nitrogen Distribution", lambda: sns.histplot(df['N'], kde=True)),
        ("Phosphorus Distribution", lambda: sns.histplot(df['P'], kde=True)),
        ("Potassium Distribution", lambda: sns.histplot(df['K'], kde=True)),
        ("Temperature Distribution", lambda: sns.histplot(df['temperature'], kde=True)),
        ("Humidity Distribution", lambda: sns.histplot(df['humidity'], kde=True)),
        ("pH Distribution", lambda: sns.histplot(df['ph'], kde=True)),
        ("Rainfall Distribution", lambda: sns.histplot(df['rainfall'], kde=True)),
        ("Temperature vs Humidity", lambda: sns.scatterplot(x='temperature', y='humidity', data=df)),
        ("pH vs Rainfall", lambda: sns.scatterplot(x='ph', y='rainfall', data=df)),
        ("Feature Correlation Heatmap", lambda: sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='coolwarm')),
    ]

    for i, (title, plot_func) in enumerate(charts, 1):
        plt.figure(figsize=(6, 4))
        plot_func()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'static/chart{i}.png')
        plt.close()


# ✅ Main route
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    nitrogen = phosphorus = potassium = temperature = humidity = ph = rainfall = ""

    if request.method == 'POST':
        try:
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]
            prediction_text = f"✅ Recommended Crop: {predicted_crop}"

        except ValueError:
            prediction_text = "⚠️ Please enter valid numbers only."

    generate_charts()
    return render_template("index.html", prediction_text=prediction_text)
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

