from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core import run_model

app = FastAPI(title="TTS API")

class TextInput(BaseModel):
    text: str

@app.post("/synthesize/")
async def synthesize(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Not Found Text")
    
    status = run_model(input_data.text)
    return {"status": status, "output_file": "output.wav"}