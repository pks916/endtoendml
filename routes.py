import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi import File, UploadFile

from src.cnnClassifier.pipeline.prediction import Prediction

IMAGE_DIR = "images"


app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    
    contents = await file.read()
    with open(f"{IMAGE_DIR}/{file.filename}", "wb") as f:
        f.write(contents)
    
    prediction = Prediction(f'{IMAGE_DIR}/{file.filename}').predict()
    os.remove(f"{IMAGE_DIR}/{file.filename}")
    return JSONResponse(prediction)


if __name__ == "__main__":
    app.run(app, host='0.0.0.0')