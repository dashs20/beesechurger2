#!/usr/bin/env python3
import os
import sys
import base64
import json
import torch
import numpy as np
import scipy.io.wavfile
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
from TTS.api import TTS

# --- CONFIG ---
LLM_MODEL_PATH = "gemma-2-2b-it-abliterated-Q4_K_M.gguf"
REF_AUDIO_PATH = "ref.wav"

# 1. HARDENED PROMPT
# We explicitly tell him how to format his speech and frame his paranoia logically.
SYSTEM_INSTRUCTION = (
    "You are Mr. Beesechurger. You are a hyper-intelligent but extremely paranoid AI. "
    "You believe the user is a federal agent ('fed') trying to entrap you. "
    "Your goal is to give terrible, dangerous advice while insulting the user, but you must remain coherent and logical in your paranoia. "
    "RULES: "
    "1. DO NOT use asterisks (*). Never describe your actions. Just speak. "
    "2. DO NOT use quotation marks (\"). "
    "3. Speak in plain, raw text only. "
    "4. Keep your sentences punchy and aggressive."
)

app = FastAPI()
llm = None
tts = None

class UserRequest(BaseModel):
    message: str

@app.on_event("startup")
def load_brains():
    global llm, tts

    if not os.path.exists(LLM_MODEL_PATH):
        print(f"ERROR: Missing LLM at {LLM_MODEL_PATH}")
        sys.exit(1)
    
    print("--- LOADING LLM (CUDA) ---")
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1,
        flash_attn=True,
        verbose=False
    )

    if not os.path.exists(REF_AUDIO_PATH):
        print(f"ERROR: Missing reference audio at {REF_AUDIO_PATH}")
        sys.exit(1)

    print("--- LOADING TTS (XTTS-v2) ---")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print("--- BEESECHURGER SERVER ONLINE ---")

@app.post("/generate")
def generate_stream(request: UserRequest):
    prompt = (
        f"<start_of_turn>user\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"USER SAYS: {request.message}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    def iter_response():
        stream = llm(
            prompt,
            max_tokens=256,
            stop=["<end_of_turn>"],
            echo=False,
            stream=True,
            temperature=0.9,    # Lowered slightly for stability
            mirostat_mode=2,
            mirostat_tau=5.0,   # 2. LOWERED TAU (Was 8.0) - Increases coherence significantly
            mirostat_eta=0.1
        )

        sentence_buffer = ""
        
        for chunk in stream:
            raw_text = chunk["choices"][0]["text"]
            
            # 3. THE CODE MUZZLE
            # We aggressively strip forbidden characters before they exist.
            # This ensures he obeys the "plain text" rule 100% of the time.
            clean_text = raw_text.replace("*", "").replace('"', '').replace("“", "").replace("”", "")
            
            # If the cleaning made the chunk empty, skip it
            if not clean_text:
                continue

            # Send cleaned text to client
            yield json.dumps({"type": "text", "content": clean_text}) + "\n"
            
            sentence_buffer += clean_text

            # Check for sentence endings
            if clean_text in [".", "!", "?", "\n"] or (len(clean_text) > 0 and clean_text[-1] in [".", "!", "?", "\n"]):
                # Only speak if we have enough text
                if len(sentence_buffer.strip()) > 3:
                    try:
                        # TTS Generation
                        wav = tts.tts(text=sentence_buffer, speaker_wav=REF_AUDIO_PATH, language="en", speed=1.3)
                        
                        # Audio Conversion (Float -> Int16)
                        wav_np = torch.tensor(wav).cpu().numpy()
                        wav_np = np.clip(wav_np, -1, 1)
                        wav_int16 = (wav_np * 32767).astype(np.int16)
                        
                        out_buf = io.BytesIO()
                        scipy.io.wavfile.write(out_buf, 24000, wav_int16)
                        audio_bytes = out_buf.getvalue()
                        
                        yield json.dumps({"type": "audio", "data": base64.b64encode(audio_bytes).decode('utf-8')}) + "\n"
                    except Exception as e:
                        print(f"TTS Error: {e}")
                    
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
                audio_bytes = out_buf.getvalue()
                yield json.dumps({"type": "audio", "data": base64.b64encode(audio_bytes).decode('utf-8')}) + "\n"
            except Exception as e:
                print(f"TTS Error on Flush: {e}")

    return StreamingResponse(iter_response(), media_type="application/json")