import os
import torch
import scipy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import DDIMScheduler
from custom_pipeline.external_libs.AudioDiffusion.audiodiffusion import AudioDiffusion
from custom_pipeline.external_libs.AudioDiffusion.audiodiffusion.audio_encoder import AudioEncoder

import custom_pipeline.gen_embed as gen_embed

class GenerateDiffusionAudio:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = torch.Generator(device=self.device)

        # The trained models are publicly available on Huggingface FYI. The diffusion model is the one I trained.
        self.diffusion_model = AudioDiffusion(model_id="Namerlight/clotho-conditional-audio-diffusion")
        self.diffusion_model.pipe.scheduler = DDIMScheduler()
        self.encoder = AudioEncoder.from_pretrained("teticio/audio-encoder")


    def gen_audio(self, timestamp: str, embeddings_input: np.ndarray, save_loc: str = None) -> str:

        embeddings_input = embeddings_input.unsqueeze_(0)
        image, (sample_rate, audio) = self.diffusion_model.generate_spectrogram_and_audio(
            generator=self.generator, encoding=embeddings_input)

        output_audio = audio
        loop = AudioDiffusion.loop_it(audio, sample_rate)
        if loop is not None:
            output_audio = loop


        savepath = os.path.join("output", "inference_testing_output", f"{timestamp}.wav")

        if save_loc:
            savepath = save_loc

        try:
            scipy.io.wavfile.write(filename=savepath, rate=sample_rate, data=output_audio)
        except FileNotFoundError as e:
            savepath = os.path.join("custom_pipeline", "output", "inference_testing_output", f"{timestamp}.wav")
            scipy.io.wavfile.write(filename=savepath, rate=sample_rate, data=output_audio)

        return savepath


# embeds_generator = gen_embed.GenerateMultimodalEmbeds()
# audgen = GenerateDiffusionAudio()
#
# image_paths = [os.path.join("data", "inference_testing_images", "sample_bus.png"),
#                os.path.join("data", "inference_testing_images", "sample_concert.png")
#                ]
# _, embds = embeds_generator.generate_embeddings(image_paths=image_paths)
#
# for idx, _ in enumerate(embds):
#     path_pic = image_paths[idx]
#     image = Image.open(path_pic)
#     plt.imshow(image)
#     plt.show()
#     saved_to = audgen.gen_audio(f"test_img_{idx}", embds[idx])
#     print(saved_to)
