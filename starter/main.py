import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_model

# Heroku DVC pull on startup.
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether income exceeds $50K/yr based on census data.",
    version="1.0.0",
)

# Load model artifacts.
model_dir = os.path.join(os.path.dirname(__file__), "model")
model = load_model(os.path.join(model_dir, "trained_model.pkl"))
encoder = load_model(os.path.join(model_dir, "encoder.pkl"))
lb = load_model(os.path.join(model_dir, "lb.pkl"))

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusData(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData):
    input_dict = data.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    X, _, _, _ = process_data(
        input_df,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X)
    prediction = lb.inverse_transform(preds)[0]
    return {"prediction": prediction}
