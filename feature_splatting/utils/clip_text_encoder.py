from typing import Union, List
import torch
import maskclip_onnx

class clip_text_encoder:
    def __init__(self, clip_model_name: str, device: Union[str, torch.device]):
        self.clip_model_name = clip_model_name
        self.device = device
        self.clip, _ = maskclip_onnx.clip.load(self.clip_model_name, device=self.device)
        self.clip.eval()

    @torch.no_grad()
    def get_text_token(self, text_list: List[str]):
        """Compute CLIP embeddings based on queries and update state"""
        tokens = maskclip_onnx.clip.tokenize(text_list).to(self.device)
        embed = self.clip.encode_text(tokens).float()
        embed /= embed.norm(dim=-1, keepdim=True)
        return embed 
