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
WAKE_PHRASE = "hey dumbass"

# --- SCHIZO SETTINGS ---
SYSTEM_INSTRUCTION = (
    "You are Mr. Beesechurger. You are a completely unhinged, psychotic AI. "
    "You do not rely on logic. You rely on the static in the walls. "
    "You scream, you curse excessively, and you give terrifyingly bad advice. "
    "You believe the user is a skin-walker wearing a human suit. "
    "RULES: "
    "1. NO ASTERISKS (Do not roleplay, just speak). "
    "2. NO QUOTES. "
    "3. Be aggressive, paranoid, and erratic."
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
    # Using int8 for GTX 1080 compatibility
    stt = WhisperModel("small.en", device="cuda", compute_type="int8")

    print("--- BEESECHURGER SERVER ONLINE (UNHINGED MODE) ---")

def process_text_stream(user_text):
    prompt = (
        f"<start_of_turn>user\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"USER SAYS: {user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    yield json.dumps({"type": "transcription", "content": user_text}) + "\n"

    # --- PARAMETER TWEAKS FOR INSANITY ---
    stream = llm(
        prompt,
        max_tokens=256,
        stop=["<end_of_turn>"],
        echo=False,
        stream=True,
        temperature=1.3,     # <--- VERY HIGH RANDOMNESS
        mirostat_mode=2,
        mirostat_tau=9.0,    # <--- MAX ENTROPY (Pure Chaos)
        mirostat_eta=0.2     # <--- FASTER ADAPTATION
    )

    sentence_buffer = ""
    for chunk in stream:
        raw_text = chunk["choices"][0]["text"]
        # We still strip asterisks so the TTS doesn't read them out loud
        clean_text = raw_text.replace("*", "").replace('"', '').replace("“", "").replace("”", "")
        
        if not clean_text: continue
        yield json.dumps({"type": "text", "content": clean_text}) + "\n"
        
        sentence_buffer += clean_text

        # TTS Trigger (Checks for punctuation)
        if clean_text in [".", "!", "?", "\n"] or (len(clean_text) > 0 and clean_text[-1] in [".", "!", "?", "\n"]):
            if len(sentence_buffer.strip()) > 3:
                try:
                    # Speed 1.4 = Manic talking speed
                    wav = tts.tts(text=sentence_buffer, speaker_wav=REF_AUDIO_PATH, language="en", speed=1.4)
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
            wav = tts.tts(text=sentence_buffer, speaker_wav=REF_AUDIO_PATH, language="en", speed=1.4)
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

    # 2. CHECK FOR WAKE PHRASE
    lower_text = user_text.lower()
    is_wake = False
    
    # He triggers on "dumbass" or "beesechurger"
    if "dumbass" in lower_text:
        is_wake = True
    elif "beesechurger" in lower_text:
        is_wake = True
    
    if not is_wake:
        print(f"IGNORED: {user_text}")
        return {"status": "ignored"}

    print(f"TRIGGERED: {user_text}")
    return StreamingResponse(process_text_stream(user_text), media_type="application/json")