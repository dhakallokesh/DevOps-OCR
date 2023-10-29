import os
import requests
import urllib.request
from datetime import datetime
from fastapi import FastAPI
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

app = FastAPI()
model = ocr_predictor('db_resnet50', "crnn_vgg16_bn", pretrained=True)

def ocr_run(imgurl: str):
    base_name = os.path.basename(imgurl)
    file_name_with_extension = base_name.split('?')[0]
    file_name, file_extension = os.path.splitext(file_name_with_extension)
    final_filename = file_name + file_extension

    urllib.request.urlretrieve(imgurl, final_filename)

    single_img_doc = DocumentFile.from_images(final_filename)
    result = model(single_img_doc)
    response_text = result.render()
    os.remove(final_filename)

    return response_text

@app.get("/")
def read_root():
    return "OCR Service"

@app.post("/extract-text/")
async def extract_text(imgurl: str):
    response_text = ocr_run(imgurl)
    return response_text

@app.post("/uploadfile/")
async def create_upload_file(imgurl: str):
    print("Input: ", imgurl)
    gptURL = "http://172.16.16.54:8080/async_gpt/"
    
    response_text = ocr_run(imgurl)
    r = requests.post(url = gptURL, json={'ocr_text':response_text})
     
    data = r.json()
    print("Output: ", data)

    return data

