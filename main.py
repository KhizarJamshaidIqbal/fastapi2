from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib

app = FastAPI()

# Load preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
classifier = joblib.load('classifier.pkl')

# Define the input data model
class StudentData(BaseModel):
    Student_ID: str
    Student_Age: int
    Sex: str
    High_School_Type: str
    Scholarship: str
    Additional_Work: str
    Attendance: str
    Reading: str
    Notes: str
    Listening_in_Class: str
    Project_work: str
    Weekly_Study_Hours: Optional[int] = None
    Transportation: Optional[str] = None
    Sports_activity: Optional[str] = None

@app.post('/predict')
async def predict(data: StudentData):
    # Convert input data to DataFrame
    df = pd.DataFrame([data.dict()])

    # Fill missing columns with default values
    if 'Weekly_Study_Hours' not in df.columns:
        df['Weekly_Study_Hours'] = 0  # or another appropriate default value
    if 'Transportation' not in df.columns:
        df['Transportation'] = 'None'  # or another appropriate default value
    if 'Sports_activity' not in df.columns:
        df['Sports_activity'] = 'No'  # or another appropriate default value

    # Ensure correct column order
    df = df[['Student_ID', 'Student_Age', 'Sex', 'High_School_Type', 'Scholarship', 'Additional_Work', 'Attendance', 'Reading', 'Notes', 'Listening_in_Class', 'Project_work', 'Weekly_Study_Hours', 'Transportation', 'Sports_activity']]

    # Transform the data using the preprocessor
    X = preprocessor.transform(df)

    # Predict using the classifier
    prediction = classifier.predict(X)

    # Return the prediction result
    return {'prediction': prediction[0]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)
