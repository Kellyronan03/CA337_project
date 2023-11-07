from flask import Flask
from flask import render_template
import pickle

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    ''' home page of app '''
    return render_template('index.html')

@app.route('/predict')
def predict():
    ''' load a model and vocab and apply it to a sample review '''
    model = pickle.load(open('ca337-nb1000-model.pkl','rb'))
    features = pickle.load(open('ca337-nb1000-features.pkl','rb'))
    review = 'I love it'
    prediction = model.predict(features.transform([review]))[0]
    return render_template('predict.html',result={'text':review,'prediction':prediction})

app.run(host='0.0.0.0', port=5000)
