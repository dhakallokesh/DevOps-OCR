import os
import requests
from datetime import datetime
from typing import Union

from fastapi import FastAPI, File, UploadFile
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

app = FastAPI()
model = ocr_predictor('db_resnet50', "crnn_vgg16_bn", pretrained=True)

@app.get("/")
def read_root():
    return "OCR Service"

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # api-endpoint
    URL = "https://ag-sales-bot.onrender.com/async_gpt/"
    now = datetime.now()     

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m%Y%H%M%S")
    print("date and time =", dt_string)

    ext_ = file.filename.split(".")[1]    
    file_location = dt_string+"."+ext_
    
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    single_img_doc = DocumentFile.from_images(file_location)
    result = model(single_img_doc)
    response_text = result.render()
    os.remove(file_location)
     
    # sending post request and saving the response as response object
    r = requests.post(url = URL, json={'ocr_text':response_text})
     
    # extracting data in json format
    data = r.json()
    return data

