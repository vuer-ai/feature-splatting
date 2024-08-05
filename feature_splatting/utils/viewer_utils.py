from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from torchtyping import TensorType
from nerfstudio.utils.rich_utils import CONSOLE

import torch

class ViewerUtils:
    def __init__(self, text_encoding_func, softmax_temp: float = 0.05, canonical_words: str = 'object'):
        self.text_encoding_func = text_encoding_func
        self.text_embedding_dict = {}
        self.update_text_embedding('canonical', canonical_words)
        self.softmax_temp = softmax_temp
        self.pca_proj = None

    @torch.no_grad()
    def update_text_embedding(self, name_key: str, raw_text: str):
        """Compute CLIP embeddings based on queries and update state"""
        texts = [x.strip() for x in raw_text.split(",") if x.strip()]
        if not texts:
            self.text_embedding_dict[name_key] = (texts, None)
        else:
            # Embed text queries
            embed = self.text_encoding_func(texts)
            self.text_embedding_dict[name_key] = (texts, embed)

    def is_embed_valid(self, name_key: str) -> bool:
        return name_key in self.text_embedding_dict and self.text_embedding_dict[name_key][1] is not None
    
    def get_text_embed(self, name_key: str) -> Optional[torch.Tensor]:
        return self.text_embedding_dict[name_key][1]
    
    def get_embed_shape(self, name_key: str) -> Optional[Tuple[int]]:
        embed = self.get_text_embed(name_key)
        if embed is not None:
            return embed.shape
        return None

    def update_softmax_temp(self, temp: float):
        self.softmax_temp = temp

    def reset_pca_proj(self):
        self.pca_proj = None
        CONSOLE.print("Reset PCA projection")

def apply_pca_colormap_return_proj(
    image: TensorType["bs":..., "d"],
    proj_V: Optional[TensorType] = None,
    low_rank_min: Optional[TensorType] = None,
    low_rank_max: Optional[TensorType] = None,
    niter: int = 5,
) -> TensorType["bs":..., "rgb":3]:
    """Convert a multichannel image to color using PCA.

    Args:
        image: Multichannel image.
        proj_V: Projection matrix to use. If None, use torch low rank PCA.

    Returns:
        Colored PCA image of the multichannel input image.
    """
    image_flat = image.reshape(-1, image.shape[-1])

    # Modified from https://github.com/pfnet-research/distilled-feature-fields/blob/master/train.py
    if proj_V is None:
        mean = image_flat.mean(0)
        with torch.no_grad():
            U, S, V = torch.pca_lowrank(image_flat - mean, niter=niter)
        proj_V = V[:, :3]

    low_rank = image_flat @ proj_V
    if low_rank_min is None:
        low_rank_min = torch.quantile(low_rank, 0.01, dim=0)
    if low_rank_max is None:
        low_rank_max = torch.quantile(low_rank, 0.99, dim=0)

    low_rank = (low_rank - low_rank_min) / (low_rank_max - low_rank_min)
    low_rank = torch.clamp(low_rank, 0, 1)

    colored_image = low_rank.reshape(image.shape[:-1] + (3,))
    return colored_image, proj_V, low_rank_min, low_rank_max
