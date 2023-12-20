import os
import gc
import torch
import numpy as np

from custom_pipeline.external_libs.ImageBind.imagebind import data as imgbdata
from custom_pipeline.external_libs.ImageBind.imagebind.models import imagebind_model
from custom_pipeline.external_libs.ImageBind.imagebind.models.imagebind_model import ModalityType


class GenerateMultimodalEmbeds:

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tri_model = imagebind_model.imagebind_huge(pretrained=True)
        self.tri_model.eval()
        self.tri_model.to(self.device)
        self.in_memory_flag = True

    def free_mem(self):
        """
        Clears the loaded model from memory. Call this in case the GPU doesn't have enough VRAM to hold all the models.
        """
        if self.in_memory_flag:
            del self.tri_model
            gc.collect()
            torch.cuda.empty_cache()
            self.in_memory_flag = False
            print("Cleared model from memory.")

    def reload_model(self):
        """
        Call only if you called free_mem previously but need to load the model again without reloading the class.
        """
        if not self.in_memory_flag:
            self.tri_model = imagebind_model.imagebind_huge(pretrained=True)
            self.tri_model.to(self.device)
            self.tri_model.eval()
            self.in_memory_flag = True

    def generate_embeddings(self, image_paths: [str]) -> (np.ndarray, np.ndarray):
        """
        Generates embeddings using Imagebind, then projects the 1024-dim output to a 512-dim embedding for captioning
        and a 100-dim embedding for guided audio generation.

        Args:
            image_paths: list of image paths as strings.

        Returns: numpy array with embeddings.
        """

        caption_projection_layer = torch.nn.Linear(1024, 512).to(self.device)
        audio_projection_layer = torch.nn.Linear(1024, 100).to(self.device)

        inputs = {
            ModalityType.VISION: imgbdata.load_and_transform_vision_data(image_paths, self.device),
        }

        with torch.no_grad():
            embeddings = self.tri_model(inputs)

        embds = embeddings[ModalityType.VISION]
        embds_for_caption = caption_projection_layer(embds)
        embds_for_audio = audio_projection_layer(embds)

        return embds_for_caption, embds_for_audio


# image_paths = [os.path.join("data", "inference_testing_images", "sample_concert.png"),
#                os.path.join("data", "inference_testing_images", "sample_concert.png")
#                ]
# gen_embs = GenerateMultimodalEmbeds()
# embds = gen_embs.generate_embeddings(image_paths=image_paths)
# print(embds[0], embds[0].shape)
# print(embds[1], embds[1].shape)
