from flask import Flask, request, render_template
from model import predict

# Define the Flask app
app = Flask(__name__, template_folder='templates')

# Define the API endpoint for the model
@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict_iris():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    result = predict([sepal_length, sepal_width, petal_length, petal_width])
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)
