"""Faithfulness metrics of the paper (Sec. VI-A).

* CLIP-I: cosine similarity between the CLIP ViT-L/14 image embeddings of reference and translation.
* CLIP-T: a BLIP caption of the reference (optionally prefixed with the style, e.g. "cartoon style of ")
  is compared with the translated image in CLIP space.
* SSIM-S: SSIM between the grayscale reference and translation.
"""
import numpy as np
import torch
from PIL import Image

CLIP_MODEL = 'openai/clip-vit-large-patch14'
BLIP_MODEL = 'Salesforce/blip-image-captioning-large'


class FaithfulnessMetrics:
    def __init__(self, device=None, clip_model=CLIP_MODEL, blip_model=BLIP_MODEL, use_clip_t=True):
        from transformers import (BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip = CLIPModel.from_pretrained(clip_model).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.blip, self.blip_processor = None, None
        if use_clip_t:
            self.blip = BlipForConditionalGeneration.from_pretrained(blip_model).to(self.device).eval()
            self.blip_processor = BlipProcessor.from_pretrained(blip_model)

    @torch.no_grad()
    def caption(self, image):
        inputs = self.blip_processor(images=image, return_tensors='pt').to(self.device)
        return self.blip_processor.decode(self.blip.generate(**inputs)[0], skip_special_tokens=True)

    @torch.no_grad()
    def image_features(self, images):
        inputs = self.clip_processor(images=images, return_tensors='pt').to(self.device)
        feats = self.clip.get_image_features(**inputs)
        feats = getattr(feats, 'image_embeds', feats)  # newer transformers may return a model output
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.no_grad()
    def text_features(self, texts):
        inputs = self.clip_processor(text=texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        feats = self.clip.get_text_features(**inputs)
        feats = getattr(feats, 'text_embeds', feats)
        return torch.nn.functional.normalize(feats, dim=-1)

    @staticmethod
    def ssim(reference, translated):
        from skimage.metrics import structural_similarity

        ref = np.asarray(reference.convert('L'))
        out = np.asarray(translated.convert('L').resize(reference.size, Image.BICUBIC))
        return float(structural_similarity(ref, out, data_range=255))

    def __call__(self, reference, translated, style_prefix=''):
        reference, translated = reference.convert('RGB'), translated.convert('RGB')
        feats = self.image_features([reference, translated])
        scores = {'clip_i': float((feats[0] * feats[1]).sum()), 'ssim': self.ssim(reference, translated)}
        if self.blip is not None:
            caption = style_prefix + self.caption(reference)
            scores['clip_t'] = float((self.text_features([caption])[0] * feats[1]).sum())
            scores['caption'] = caption
        return scores
