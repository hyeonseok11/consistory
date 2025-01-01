import torch
import clip
from PIL import Image

def get_clip_score(image, text):
    """
    Calculate CLIP score for an image and text pair
    
    Args:
        image: PIL Image object or path to image file
        text: Text prompt to compare with the image
    
    Returns:
        float: CLIP score (cosine similarity between image and text embeddings)
    """
    # Load the pre-trained CLIP model
    model, preprocess = clip.load('ViT-B/32')
    
    # Handle both PIL Image and file path inputs
    if isinstance(image, str):
        image = Image.open(image)
    
    # Preprocess the image and tokenize the text
    image_input = preprocess(image).unsqueeze(0)
    text_input = clip.tokenize([text])
    
    # Move the inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_input = image_input.to(device)
    text_input = text_input.to(device)
    model = model.to(device)
    
    # Generate embeddings for the image and text
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    
    return clip_score
