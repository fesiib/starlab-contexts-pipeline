
from transformers import VideoCLIP

video_clip_model = VideoCLIP.from_pretrained("openai/clip-vit-base-patch32")

def embed_video(video):
    """
    Embed a video using a VideoCLIP model
    """