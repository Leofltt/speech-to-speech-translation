import gradio as gr
import numpy as np
import torch
from datasets import load_dataset

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor, pipeline
from transformers import BarkModel, BarkProcessor



device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load speech translation checkpoint
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# load text-to-speech checkpoint and speaker embeddings
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

barkmodel = BarkModel.from_pretrained("suno/bark")
barkprocessor = BarkProcessor.from_pretrained("suno/bark")

# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


def translate(audio):
    outputs = asr_pipe(audio, max_new_tokens=256, generate_kwargs={"task": "transcribe", "language": "it"})
    return outputs["text"]


def synthesise(text):
    inputs = barkprocessor(text=[text], voice_preset="v2/it_speaker_4",return_tensors="pt")
    speech = barkmodel.generate(**inputs, do_sample=True)
    return speech


def speech_to_speech_translation(audio):
    translated_text = translate(audio)
    synthesised_speech = synthesise(translated_text)
    synthesised_speech = (synthesised_speech.numpy() * 32767).astype(np.int16)
    return 16000, synthesised_speech


title = "Cascaded STST"
description = """
Demo for cascaded speech-to-speech translation (STST), mapping from source speech in any language to target speech in Italian. Demo uses OpenAI's [Whisper Base](https://huggingface.co/openai/whisper-base) model for speech translation, and Suno's
[Bark](https://huggingface.co/suno/bark) model for text-to-speech:

![Cascaded STST](https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/s2st_cascaded.png "Diagram of cascaded speech to speech translation")
"""

demo = gr.Blocks()

mic_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    title=title,
    description=description,
)

file_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    examples=[["./example.wav"]],
    title=title,
    description=description,
)

with demo:
    gr.TabbedInterface([mic_translate, file_translate], ["Microphone", "Audio File"])

demo.launch()
