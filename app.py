from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np


app = Flask(__name__)

model, scalar = pickle.load(open("diabetesModel.pkl", "rb"))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    text7 = request.form['7']
    text8 = request.form['8']
 
    sample_input = np.array(list(map(float, [text1,text2,text3,text4,text5,text6,text7,text8])))
    sample_input = sample_input.reshape(1,-1)
    row_df = scalar.transform(sample_input)
    prediction=model.predict(row_df)
    if prediction[0]==1:
        return render_template('result.html',pred=f'You have chance of having diabetes')
    else:
        return render_template('result.html',pred=f'You are safe')




if __name__ == '__main__':
    app.run(debug=True)
