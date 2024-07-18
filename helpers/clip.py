import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def clip_embed_video(video):
    """
    Embed a video using a CLIP model
    """
    pass

def clip_embed_image(images):
    """
    Embed an image using a CLIP model
    """
    ### transform the image to a tensor
    embeddings = []
    for image in images:
        image = preprocess(Image.open(image)).unsqueeze(0).to(device)
        ### get the image features
        with torch.no_grad():
            image_features = model.encode_image(image)
        embeddings.append(image_features)
    return embeddings


def clip_embed_text(texts):
    """
    Embed a text using a CLIP model
    """

    embeddings = []

    for text in texts:
        text = clip.tokenize([text]).to(device)
        ### get the text features
        with torch.no_grad():
            text_features = model.encode_text(text)
        embeddings.append(text_features)
    return embeddings