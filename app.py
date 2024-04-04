import gradio as gr
import numpy as np
import torch
from datasets import load_dataset

from transformers import pipeline
from transformers import BarkModel, BarkProcessor

from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

device = "cuda:0" if torch.cuda.is_available() else "cpu"

asr_model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
asr_processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

asr_model.to(device)

bark_model = BarkModel.from_pretrained("suno/bark-small")
bark_processor = BarkProcessor.from_pretrained("suno/bark-small")

bark_model.to(device)


def translate(audio):
    inputs = asr_processor(audio, sampling_rate=16000, return_tensors="pt")
    generated_ids = asr_model.generate(inputs["input_features"],attention_mask=inputs["attention_mask"], 
                                       forced_bos_token_id=asr_processor.tokenizer.lang_code_to_id['it'],)
    translation = asr_processor.batch_decode(generated_ids, skip_special_tokens=True)
    return translation


def synthesise(text):
    inputs = bark_processor(text=text, voice_preset="v2/it_speaker_4",return_tensors="pt")
    speech = bark_model.generate(**inputs, do_sample=True)
    speech = speech.cpu().numpy().squeeze()
    return speech


def speech_to_speech_translation(audio):
    translated_text = translate(audio)
    synthesised_speech = synthesise(translated_text)
    synthesised_speech = (synthesised_speech * 32767).astype(np.int16)
    return 16000, synthesised_speech


title = "Cascaded STST"
description = """
Demo for cascaded speech-to-speech translation (STST), mapping from source speech in any language to target speech in Italian. Demo uses Meta's [Speech2Text](https://huggingface.co/facebook/s2t-medium-mustc-multilingual-st) model for speech translation, and Suno's
[Bark](https://huggingface.co/suno/bark) model for text-to-speech:

![Cascaded STST](https://huggingface.co/datasets/huggingface-course/audio-course-images/resolve/main/s2st_cascaded.png "Diagram of cascaded speech to speech translation")
"""

demo = gr.Blocks()

mic_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    title=title,
    description=description,
)

file_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    examples=[["./example.wav"]],
    title=title,
    description=description,
)

with demo:
    gr.TabbedInterface([mic_translate, file_translate], ["Microphone", "Audio File"])

demo.launch()
