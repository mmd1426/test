from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch


MODEL_ID = "whisper-tiny-fa"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID).to(device)

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if device == "cuda" else -1
)

def asr(audio_path):

    result = asr(audio_path)

    return result["text"]