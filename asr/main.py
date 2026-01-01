from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import os

from core import asr

app = FastAPI(title="ASR API")

UPLOAD_DIR = "/app/voices"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/asr")
async def speech_to_text(file: UploadFile = File(...)):

    file_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{file_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = asr(file_path)

    os.remove(file_path)

    return {
        "text": text
    }
