from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Loading the Vision Encoder-Decoder model from pretrained weights
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Checking for GPU availability and moving the model to CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Setting parameters for generation
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Initializing the ViT Image Processor and Tokenizer
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Defining a function to predict captions for given image paths
def predict_step(image_paths):
    images = []
    
    # Iterating over provided image paths and loading images
    for image_path in image_paths:
        i_image = Image.open(image_path)
        
        # Converting image to RGB mode if not already in RGB
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)
    
    # Extracting pixel values from images and moving to CUDA device
    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Generating captions using the Vision Encoder-Decoder model
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decoding generated token IDs into readable captions
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    return preds
