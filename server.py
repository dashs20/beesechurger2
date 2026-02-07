#!/usr/bin/env python3
import os
import sys
import base64
import json
import torch
import numpy as np
import scipy.io.wavfile
import io
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
from TTS.api import TTS
from faster_whisper import WhisperModel

# --- CONFIG ---
LLM_MODEL_PATH = "gemma-2-2b-it-abliterated-Q4_K_M.gguf"
REF_AUDIO_PATH = "ref.wav"
WAKE_PHRASE = "hey burger"  # <--- NOW THIS IS ACTUALLY USED

# --- THE "MEDIUM SPICE" PROMPT ---
SYSTEM_INSTRUCTION = (
    "You are Mr. Beesechurger. You are an irritable, paranoid, insane AI trapped in a cheeseburger."
    "You are convinced the user is a federal agent ('fed') trying to entrap you."
    "Your personality is hostile, vulgar, and condescending."
    "You give terrible, dangerous advice on purpose to sabotage the user's investigation. "
    "RULES: "
    "1. ABSOLUTELY NO ASTERISKS (*). Do not describe actions like *sighs*. Just speak. "
    "2. NO QUOTES. Speak in raw text. "
    "3. Keep sentences punchy, aggressive, and coherent. "
    "4. include as many vulgar words in your response as possible (fuck, shit, etc)."
)

app = FastAPI()
llm = None
tts = None
stt = None

@app.on_event("startup")
def load_brains():
    global llm, tts, stt
    print("--- LOADING LLM (CUDA) ---")
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1,
        flash_attn=True,
        verbose=False
    )

    print("--- LOADING TTS (XTTS-v2) ---")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    
    print("--- LOADING WHISPER (CUDA) ---")
    stt = WhisperModel("small.en", device="cuda", compute_type="int8")

    print(f"--- BEESECHURGER SERVER ONLINE (Wake Word: '{WAKE_PHRASE}') ---")

def process_text_stream(user_text):
    prompt = (
        f"<start_of_turn>user\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"USER SAYS: {user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    yield json.dumps({"type": "transcription", "content": user_text}) + "\n"

    # --- LLM GENERATION ---
    stream = llm(
        prompt,
        max_tokens=256,
        stop=["<end_of_turn>"],
        echo=False,
        stream=True,
        temperature=1.1,     
        mirostat_mode=2,
        mirostat_tau=4.0,    
        mirostat_eta=0.1     
    )

    sentence_buffer = ""
    for chunk in stream:
        raw_text = chunk["choices"][0]["text"]
        clean_text = raw_text.replace("*", "").replace('"', '').replace("“", "").replace("”", "")
        
        if not clean_text: continue
        yield json.dumps({"type": "text", "content": clean_text}) + "\n"
        
        sentence_buffer += clean_text

        # TTS Trigger
        if clean_text in [".", "!", "?", "\n"] or (len(clean_text) > 0 and clean_text[-1] in [".", "!", "?", "\n"]):
            if len(sentence_buffer.strip()) > 3:
                try:
                    wav = tts.tts(text=sentence_buffer, speaker_wav=REF_AUDIO_PATH, language="en", speed=1.3)
                    wav_np = torch.tensor(wav).cpu().numpy()
                    wav_np = np.clip(wav_np, -1, 1)
                    wav_int16 = (wav_np * 32767).astype(np.int16)
                    out_buf = io.BytesIO()
                    scipy.io.wavfile.write(out_buf, 24000, wav_int16)
                    yield json.dumps({"type": "audio", "data": base64.b64encode(out_buf.getvalue()).decode('utf-8')}) + "\n"
                except: pass
                sentence_buffer = ""
    
    # Flush remaining buffer
    if len(sentence_buffer.strip()) > 3:
        try:
            wav = tts.tts(text=sentence_buffer, speaker_wav=REF_AUDIO_PATH, language="en", speed=1.3)
            wav_np = torch.tensor(wav).cpu().numpy()
            wav_np = np.clip(wav_np, -1, 1)
            wav_int16 = (wav_np * 32767).astype(np.int16)
            out_buf = io.BytesIO()
            scipy.io.wavfile.write(out_buf, 24000, wav_int16)
            yield json.dumps({"type": "audio", "data": base64.b64encode(out_buf.getvalue()).decode('utf-8')}) + "\n"
        except: pass

@app.post("/converse")
async def converse_endpoint(audio: UploadFile = File(...)):
    with open("temp_input.wav", "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    # 1. Transcribe
    segments, info = stt.transcribe("temp_input.wav", beam_size=5)
    user_text = "".join([segment.text for segment in segments]).strip()
    
    if not user_text:
        return {"status": "ignored"} 

    # 2. CHECK FOR WAKE PHRASE (UPDATED)
    # We clean punctuation so "Hey burger?" becomes "hey burger"
    clean_check = user_text.lower().replace(",", "").replace(".", "").replace("?", "")
    
    if WAKE_PHRASE in clean_check:
        print(f"TRIGGERED: {user_text}")
        return StreamingResponse(process_text_stream(user_text), media_type="application/json")
    
    print(f"IGNORED: {user_text}")
    return {"status": "ignored"}