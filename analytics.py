import torch
from transformers import CLIPProcessor, CLIPModel
import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from fvcore.nn import FlopCountAnalysis

from vars import labels_file

device = "cuda"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def compute_clip_score(generated_dir, prompts):
    scores = []
    files = sorted([f for f in os.listdir(generated_dir) if f.endswith(".png") or f.endswith(".jpg")])
    files.sort(key=lambda x: int(x.split(sep=".")[0].split("_")[1]))

    for fname, prompt in zip(files, prompts):
        img_path = os.path.join(generated_dir, fname)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping image {fname}: {e}")
            continue

        # Process inputs
        inputs = clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = clip_model(**inputs)

        # Normalize embeddings
        text_emb = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        img_emb = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)

        # Cosine similarity
        score = (text_emb @ img_emb.T).item()
        scores.append(score)

    return sum(scores) / len(scores)


def compute_ssim(real_dir, generated_dir):
    files_real = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    files_gen = sorted([f for f in os.listdir(generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    scores = []
    for fr, fg in zip(files_real, files_gen):
        try:
            img_real = np.array(Image.open(os.path.join(real_dir, fr)).convert("L").resize((256, 256)))
            img_gen = np.array(Image.open(os.path.join(generated_dir, fg)).convert("L").resize((256, 256)))
        except Exception as e:
            print(f"Skipping {fr} and {fg} due to error: {e}")
            continue

        # ✅ move SSIM calculation outside the except
        score = ssim(img_real, img_gen, data_range=255)
        scores.append(score)

    return np.mean(scores) if scores else 0


class SDForwardWrapper(torch.nn.Module):
    def __init__(self, pipe):
        super().__init__()
        self.clip = pipe.text_encoder
        self.unet = pipe.unet
        self.vae = pipe.vae

    def forward(self, input_ids, latent):
        hidden_states = self.clip.text_model.embeddings(input_ids)
        for block in self.clip.text_model.encoder.layers:
            hidden_states = block(hidden_states, attention_mask=attention_mask, causal_attention_mask=None)[0]

        t = torch.tensor([1], device=latent.device)
        noise_pred = self.unet(latent, t, encoder_hidden_states=hidden_states)

        output = noise_pred.sample if hasattr(noise_pred, "sample") else noise_pred
        return output


MAX_TO = 59

with open(labels_file, "r") as file:
    labels = file.readlines()

labels = [label.replace("\n", "") for label in labels]

labels = labels[:MAX_TO]
clip_score = compute_clip_score("data/deepcached", labels)
print("Average CLIP Score:", clip_score)

real_dir = "data/baseline"
gen_dir = "data/deepcached"

ssim_score = compute_ssim(real_dir, gen_dir)
print("Average SSIM:", ssim_score)
