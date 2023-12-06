from flask import Flask, render_template, request
import pickle
import re
from nltk.tokenize import word_tokenize

app = Flask(__name__)

def custom_preprocessor(text):
    # Same function as defined in the Jupyter Notebook
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    processed_text = ' '.join(tokens)
    return processed_text

models = {
    'model_nb': pickle.load(open('ca337-model_nb-model.pkl', 'rb')),
    'model_lr': pickle.load(open('ca337-model_lr-model.pkl', 'rb')),
    'model_rf': pickle.load(open('ca337-model_rf-model.pkl', 'rb')),
}
features = pickle.load(open('ca337-nb1000-features.pkl', 'rb'))


model_names = {
    'model_nb': 'Naive Bayes',
    'model_lr': 'Logistic Regression',
    'model_rf': 'Random Forest'
}
@app.route('/')
@app.route('/index')
def index():
    ''' home page of app '''
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the review from the form
        review = request.form['review_text']
        selected_model_id = request.form['model_selection']
        selected_model_name = model_names.get(selected_model_id, "Unknown Model")
        model_to_use = models[selected_model_id]
        prediction = model_to_use.predict(features.transform([review]))[0]
        
    else:
        # Default values for GET request
        review = 'I loved it'
        selected_model_id = list(models.keys())[0]
        selected_model_name = model_names.get(selected_model_id, "Unknown Model")
        model_to_use = models[selected_model_id]
        prediction = model_to_use.predict(features.transform([review]))[0]
        
    # Convert numerical prediction to string
    prediction_text = 'Positive' if prediction == 1 else 'Negative'
    return render_template('predict.html', result={'text': review, 'prediction': prediction_text, 'model_used': selected_model_name})


app.run(host='0.0.0.0', port=8000)
