import numpy as np

import custom_pipeline.gen_embed as gen_embed
import custom_pipeline.gen_audio as gen_audio
import custom_pipeline.gen_caption as gen_caption


class ImagebindDiffusionSynthesis:

    def __init__(self):
        self.embeds_generator = gen_embed.GenerateMultimodalEmbeds()
        self.audio_generator = gen_audio.GenerateDiffusionAudio()
        self.caption_generator = gen_caption.GenerateDeCapCaptions(saved_features=False)
        print("All Models initialized.")
        self.multi_embeddings = None

    def gen_embeds(self, input_path: str) -> (np.ndarray, np.ndarray):
        self.multi_embeddings = self.embeds_generator.generate_embeddings([input_path])

    def gen_caption(self) -> str:
        cap_op = self.caption_generator.generate_captions(embeddings_input=self.multi_embeddings[0])
        return cap_op

    def gen_audio(self, timestamp, save_path=None) -> str:
        aud_op_path = self.audio_generator.gen_audio(timestamp, self.multi_embeddings[1], save_path)
        return aud_op_path
