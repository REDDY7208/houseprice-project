from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
with open('HousePricePred', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # If the form is submitted using POST, process the data
        features = [float(request.form['feature1']), float(request.form['feature2']), float(request.form['feature3']),float(request.form['feature4'])]

        # Make a prediction using the loaded model
        prediction = model.predict([features])[0]

        return render_template('result.html', prediction=prediction)

    # If the request is a GET request or the form hasn't been submitted yet
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
