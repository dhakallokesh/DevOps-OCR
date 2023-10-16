import os
import requests
import urllib.request
from datetime import datetime
from fastapi import FastAPI
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

app = FastAPI()
model = ocr_predictor('db_resnet50', "crnn_vgg16_bn", pretrained=True)

@app.get("/")
def read_root():
    return "OCR Service"

@app.post("/uploadfile/")
async def create_upload_file(imgurl: str):
    # api-endpoint
    gptURL = "https://ag-sales-bot.onrender.com/async_gpt/"
    now = datetime.now()     

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d%m%Y%H%M%S")
    print("date and time =", dt_string)

    base_name = os.path.basename(imgurl)
    file_name_with_extension = base_name.split('?')[0]
    file_name, file_extension = os.path.splitext(file_name_with_extension)
    final_filename = file_name + file_extension
    print(final_filename)

    urllib.request.urlretrieve(imgurl, final_filename)

    # ext_ = file.filename.split(".")[1]    
    # file_location = dt_string+"."+ext_
    
    # with open(file_location, "wb+") as file_object:
    #     file_object.write(file.file.read())
    
    single_img_doc = DocumentFile.from_images(final_filename)
    result = model(single_img_doc)
    response_text = result.render()
    os.remove(final_filename)
     
    # sending post request and saving the response as response object
    r = requests.post(url = gptURL, json={'ocr_text':response_text})
     
    # extracting data in json format
    data = r.json()
    return data

