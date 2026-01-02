from fastapi import FastAPI, Response, HTTPException
import io
import wave
from piper import PiperVoice
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# تنظیم لاگ‌ها
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# مسیر مدل
model_path = "/app/model/Mana-Persian-Piper/fa_IR-mana-medium.onnx"

try:
    voice = PiperVoice.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"FAILED to load model: {e}")
    voice = None

app = FastAPI()
executor = ThreadPoolExecutor()


def generate_audio_sync(text: str) -> bytes:
    if voice is None:
        raise Exception("Voice model is not loaded.")
        
    buf = io.BytesIO()
    
    # دریافت جریان داده‌ها (که شامل آبجکت‌های AudioChunk است)
    audio_stream = voice.synthesize(text)
    
    # استخراج بایت‌های خام از درون آبجکت‌ها
    raw_chunks = []
    for chunk in audio_stream:
        # بررسی می‌کنیم داده‌ها در کدام متغیر ذخیره شده‌اند
        if hasattr(chunk, "bytes"):
            raw_chunks.append(chunk.bytes)
        elif hasattr(chunk, "data"):
            raw_chunks.append(chunk.data)
        elif hasattr(chunk, "audio"):
            raw_chunks.append(chunk.audio)
        else:
            # جهت دیباگ اگر نام متغیر چیز دیگری بود
            logger.error(f"Unknown chunk structure. Attributes: {dir(chunk)}")
            raise ValueError("Could not find raw bytes in AudioChunk")

    raw_data = b"".join(raw_chunks)
    
    # نوشتن فایل WAV
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        wav_file.writeframes(raw_data)
    
    return buf.getvalue()

@app.post("/v1/audio/speech")
async def speech(req: dict):
    try:
        text = req.get("input", "")
        logger.info(f"Received request for text length: {len(text)}")

        if not text:
            return Response(status_code=400, content="Empty text input")

        loop = asyncio.get_running_loop()
        audio_content = await loop.run_in_executor(executor, generate_audio_sync, text)

        logger.info(f"Generated audio size: {len(audio_content)} bytes")
        
        return Response(
            content=audio_content, 
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=output.wav"}
        )

    except Exception as e:
        logger.error(f"Error generating audio: {e}", exc_info=True)
        return Response(status_code=500, content=str(e))