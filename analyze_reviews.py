from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load the sentiment analysis model
model = load("k_neighbors.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = model.predict([review])[0]
        result = "Positive" if sentiment == 1 else "Negative"
        return render_template('result.html', review=review, sentiment=result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
