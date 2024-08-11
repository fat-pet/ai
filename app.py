from flask import Flask, request
import pandas as pd
import xgboost as xgb

app = Flask(__name__)


@app.route("/api/predict", methods=["GET"])
def predict():
    age = request.args.get("age")
    sex = request.args.get("sex")
    weight = request.args.get("weight")
    neck_size = request.args.get("neck_circ")
    chest = request.args.get("chest_circ")
    breed = request.args.get("breed")
    species = request.args.get("species")

    model = xgb.Booster()
    model.load_model("xgb.json")

    data = pd.DataFrame(
        {
            "breed": [int(breed)],
            "age": [float(age)],
            "sex": [int(sex)],
            "weight": [float(weight)],
            "neck_size": [float(neck_size)],
            "chest": [float(chest)],
            "class": [int(species)],
        }
    )

    dmatrix = xgb.DMatrix(data)

    prediction = model.predict(dmatrix)

    return str(prediction[0])


if __name__ == "__main__":
    app.run()
