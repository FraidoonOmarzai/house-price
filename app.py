from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')


@app.route("/")
def house_price():

    return render_template("house_price.html")

@app.route('/predictHP', methods=["POST"])
def predictHP():


    if request.method == "POST":

        Location = request.form['Location']
        Rooms = request.form['Rooms']
        Type = request.form['Type']
        Postcode = request.form['Postcode']
        Distance = request.form['Distance']
        Year = request.form['Year']

        input_variables = pd.DataFrame([[Location, Rooms, Type, Postcode, Distance, Year]],
                                    columns=['Suburb' , 'Rooms', 'Type', 'Postcode', 'Distance', 'Year'],
                                    dtype=float)

        model = joblib.load("Trained Model/housepriceprediction.joblib")
        prediction=model.predict(input_variables)[0]

        prediction = "Price of the house is: "+str(prediction) +"$"
        print(prediction)


    return(render_template("result.html", prediction_text=prediction))


if __name__ == '__main__':
    app.run(debug = True)
