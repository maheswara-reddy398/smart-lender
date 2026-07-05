from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model and Scaler
model = pickle.load(open("../models/rdf.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        Gender = int(request.form["Gender"])
        Married = int(request.form["Married"])
        Dependents = int(request.form["Dependents"])
        Education = int(request.form["Education"])
        Self_Employed = int(request.form["Self_Employed"])
        ApplicantIncome = float(request.form["ApplicantIncome"])
        CoapplicantIncome = float(request.form["CoapplicantIncome"])
        LoanAmount = float(request.form["LoanAmount"])
        Loan_Amount_Term = float(request.form["Loan_Amount_Term"])
        Credit_History = float(request.form["Credit_History"])
        Property_Area = int(request.form["Property_Area"])

        features = np.array([[
            Gender,
            Married,
            Dependents,
            Education,
            Self_Employed,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Property_Area
        ]])

        # Scale Input
        features = scaler.transform(features)

        # Prediction
        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Loan Approved ✅"
        else:
            result = "Loan Rejected ❌"

        return render_template("result.html", prediction=result)

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)