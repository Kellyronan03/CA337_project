from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    ''' home page of app '''
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    ''' load a model and vocab and apply it to user input review '''
    model = pickle.load(open('ca337-nb1000-model.pkl','rb'))
    features = pickle.load(open('ca337-nb1000-features.pkl','rb'))

    if request.method == 'POST':
        # Get the review from the form
        review = request.form['review_text']
    else:
        # Default review for GET request
        review = 'I loved it'
    
    prediction = model.predict(features.transform([review]))[0]
    # Convert numerical prediction to string
    prediction_text = 'Positive' if prediction == 1 else 'Negative'
    return render_template('predict.html', result={'text': review, 'prediction': prediction_text})


app.run(host='0.0.0.0', port=8000)
