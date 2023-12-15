import os
import scipy
import torch
from PIL import Image
from diffusers import AudioLDM2Pipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

class pretrained_caption_audsynth:

    def __init__(self):
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
        self.audio_model = AudioLDM2Pipeline.from_pretrained(pretrained_model_name_or_path="cvssp/audioldm2", torch_dtype=torch.float16).to("cuda")
        self.timestamp = None

    def generate_captions(self, raw_image):
        self.timestamp = time.time()
        inputs = self.caption_processor(raw_image, return_tensors="pt").to("cuda")
        out = self.caption_model.generate(**inputs)
        return self.caption_processor.decode(out[0], skip_special_tokens=True)

    def generate_audio(self, prompt_input, seed=None):
        generator = torch.Generator("cuda").manual_seed(0)
        audio = self.audio_model(
            prompt_input,
            num_inference_steps=200,
            audio_length_in_s=10.0,
            num_waveforms_per_prompt=3,
        ).audios

        savepath = os.path.join("..", "output", "inference_testing_output", f"{self.timestamp}.wav")

        try:
            scipy.io.wavfile.write(filename=savepath, rate=16000, data=audio[0])
        except FileNotFoundError as e:
            savepath = os.path.join("existing_pipeline", "output", "inference_testing_output", f"{self.timestamp}.wav")
            scipy.io.wavfile.write(filename=savepath, rate=16000, data=audio[0])

        return savepath


# inp_img = Image.open(fp=os.path.join("..", "data", "inference_testing_images", "sample_concert.png"))
# capgen = pretrained_caption_audsynth()
# caption = capgen.generate_captions(raw_image=inp_img)
# audio = capgen.generate_audio(prompt_input=caption)