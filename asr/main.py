from fastapi import FastAPI, UploadFile, File, Form
from transformers import pipeline
import io
import soundfile as sf

app = FastAPI()

asr = pipeline(
    "automatic-speech-recognition",
    model="/app/model/whisper-tiny-fa"
)

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(...)
):
    audio_bytes = await file.read()

    audio, sr = sf.read(io.BytesIO(audio_bytes))
    text = asr({"array": audio, "sampling_rate": sr})["text"]

    return {"text": text}
