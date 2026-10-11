from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from typing import Literal
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ml_model = {} #{"model":"credit_risk_model.pkl"}



@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model['model'] = joblib.load('credit_risk_model.pkl')
    ml_model['threshold'] = joblib.load('best_threshold.pkl')

    yield

    ml_model.clear()

app = FastAPI(lifespan=lifespan)


#The only columns that user will see and provide inputs.
class LoanApplication(BaseModel): #Pydantic Model (Validation)
    # Bounds reject impossible inputs only (same rules as the web form); values outside the
    # training range but still possible are accepted, and categories are the training categories.
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: Literal["MORTGAGE", "OTHER", "OWN", "RENT"]
    person_emp_length: float = Field(ge=0, le=60)
    loan_intent: Literal["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(gt=0, le=100)
    loan_percent_income: float = Field(ge=0, le=10)
    cb_person_default_on_file: Literal["N", "Y"]
    cb_person_cred_hist_length: int = Field(ge=0, le=80)

    @model_validator(mode="after")
    def emp_length_within_age(self):
        if self.person_emp_length > self.person_age:
            raise ValueError("person_emp_length cannot exceed person_age")
        return self



@app.post('/predict')
def predict(data : LoanApplication):
    input_df = pd.DataFrame([data.dict()])

    probability = ml_model['model'].predict_proba(input_df)[:, 1][0]

    prediction = int(probability >= ml_model["threshold"])

    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": ml_model["threshold"],
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }

app.mount("/", StaticFiles(directory="static", html=True), name="static")
