import os
import gc
import json
import pickle
import random
import numpy as np
from tqdm import tqdm
from enum import Enum
from typing import Tuple
import PIL.Image as Image
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel

import torch
from torch import nn

import clip
from clip.simple_tokenizer import SimpleTokenizer

import custom_pipeline.gen_embed as gen_embed

"""
This file is very heavily based (basically entirely) on the example notebook at https://github.com/dhg-wei/DeCap
Original credits to Wei Li and Linchao Zhu, Longyin Wen and Yi Yang.
"""

device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=torch.device(device), jit=False)
tokenizer = clip.tokenize
_Tokenizer = SimpleTokenizer()

data_path = os.path.join("custom_pipeline", "pretrained_weights", "DeCap_CoCo")

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class DeCap(nn.Module):

    def __init__(self, prefix_size: int = 512):

        self.config_path = os.path.join(data_path, 'decoder_config.pkl')

        super(DeCap, self).__init__()
        with open(self.config_path, 'rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, self.embedding_size))

    def forward(self, clip_features, tokens):
        embedding_text = self.decoder.transformer.wte(tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1, 1, self.embedding_size)
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out


def Decoding(model, clip_features):
    model.eval()
    embedding_cat = model.clip_project(clip_features).reshape(1, 1, -1)
    entry_length = 30
    temperature = 1
    tokens = None
    for i in range(entry_length):
        outputs = model.decoder(inputs_embeds=embedding_cat)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits_max = logits.max()
        logits = torch.nn.functional.softmax(logits)
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = model.decoder.transformer.wte(next_token)

        if tokens is None:
            tokens = next_token

        else:
            tokens = torch.cat((tokens, next_token), dim=1)
        if next_token.item() == 49407:
            break
        embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)
    try:
        output_list = list(tokens.squeeze().cpu().numpy())
        output = _Tokenizer.decode(output_list)
    except:
        output = 'None'
    return output

class GenerateDeCapCaptions():

    def __init__(self, saved_features=False):
        self.weights_path = os.path.join(data_path, "coco_prefix-009.pt")
        self.model = DeCap()
        self.model.load_state_dict(torch.load(self.weights_path, map_location=torch.device('cpu')), strict=False)
        self.model.to(device)
        self.model.eval()
        self.in_memory_flag = True

        self.text_features = []
        if not saved_features:
            print("Computing Text Features")
            self.get_text_features()
            torch.save(self.text_features, f=os.path.join(data_path, 'text_features.pt'))
        if saved_features:
            print("Loading Text Features")
            torch.load(torch.load(f=os.path.join(data_path, 'text_features.pt')))

    def free_mem(self):
        """
        Clears the loaded model from memory. Call this in case the GPU doesn't have enough VRAM to hold all the models.
        """
        if self.in_memory_flag:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.in_memory_flag = False
            print("Cleared model from memory.")

    def reload_model(self):
        """
        Call only if you called free_mem previously but need to load the model again without reloading the class.
        """
        if not self.in_memory_flag:
            self.model = DeCap()
            self.model.load_state_dict(torch.load(self.weights_path, map_location=torch.device('cpu')), strict=False)
            self.model.to(device)
            self.model.eval()
            self.in_memory_flag = True


    def get_text_features(self):
        """
        Gets features for the text decoder based on the CLIP model and Coco labels.
        """
        with open(os.path.join(data_path, "coco_train.json"), 'r') as f:
            data = json.load(f)

        data = random.sample(data, 10000)
        captions = []
        batch_size = 1000
        clip_model.eval()

        for i in tqdm(range(0, len(data[:]) // batch_size)):
            texts = data[i * batch_size:(i + 1) * batch_size]
            with torch.no_grad():
                texts_token = tokenizer(texts).to(device)
                text_feature = clip_model.encode_text(texts_token)
                self.text_features.append(text_feature)
                captions.extend(texts)

        self.text_features = torch.cat(self.text_features, dim=0)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True).float()

    def generate_captions(self, embeddings_input: np.ndarray) -> str:
        """
        Generate captions for a single input image by passing its embeddings (either CLIP or from gen_embed

        Args:
            embeddings_input: embedded image

        Returns: generated image caption as a string

        """
        with torch.no_grad():
            sim = embeddings_input @ self.text_features.T.float()
            sim = (sim * 100).softmax(dim=-1)
            prefix_embedding = sim @ self.text_features.float()
            prefix_embedding /= prefix_embedding.norm(dim=-1, keepdim=True)
            generated_text = Decoding(self.model, prefix_embedding)
            generated_text = generated_text.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
            return generated_text


# embeds_generator = gen_embed.GenerateMultimodalEmbeds()
# captioner = GenerateDeCapCaptions()
# image_paths = [os.path.join("data", "inference_testing_images", "sample_bus.png"),
#                os.path.join("data", "inference_testing_images", "sample_concert.png")
#                ]
# embds, _ = embeds_generator.generate_embeddings(image_paths=image_paths)
#
# for idx, _ in enumerate(embds):
#     path_pic = image_paths[idx]
#     image = Image.open(path_pic)
#     plt.imshow(image)
#     plt.show()
#     caption = (captioner.generate_captions(embeddings_input=embds[idx]))
#     print(caption)
