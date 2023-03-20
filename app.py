import pickle
from flask import Flask, request, render_template
import pandas as pd

from sklearn.preprocessing import StandardScaler
from source.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Create route for home page


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=request.form("gender"),
            rece_ethnicity=request.form("race_ethnicity"),
            parental_level_of_education=request.form("parental_level_of_education"),
            lunch=request.form("lunch"),
            test_preparation_course=request.form("test_preparation_course"),
            reading_score=request.form("reading_score"),
            writing_score=request.form("writing_score"),
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template("home.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
