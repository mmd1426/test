import wave
from piper import PiperVoice

voice = PiperVoice.load("/app/model/Mana-Persian-Piper/fa_IR-mana-medium.onnx")

def run_model(text):
    
    with wave.open("/app/voices/output.wav", "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    
    return "Ok"
