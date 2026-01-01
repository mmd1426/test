import wave
from piper import PiperVoice

voice = PiperVoice.load("/model/fa_IR-mana-medium.onnx")

def run_model(text):
    
    with wave.open("voices/output.wav", "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    
    return "Ok"