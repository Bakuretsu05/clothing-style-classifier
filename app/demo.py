import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image
import gradio as gr
import numpy as np
from pathlib import Path
import os

# Configuration
NUM_CLASSES = 7
INPUT_SIZE = 224
MODEL_PATH = 'models/best_model.pth'

# Class labels (must match training script)
CLASS_NAMES = [
    'T-shirt',
    'Polo',
    'Formal_Shirt',
    'Tank_Top',
    'Sweater',
    'Hoodie',
    'Jacket'
]


def create_model(num_classes=NUM_CLASSES):
    """Create MobileNetV3-Small model architecture."""
    model = models.mobilenet_v3_small(weights=None)  # No pretrained weights needed for inference
    
    # Replace classifier head (must match training architecture)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    return model


def load_model(model_path=MODEL_PATH):
    """Load the trained PyTorch model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please train the model first using train.py"
        )
    
    # Create model
    model = create_model()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    # Get class names from checkpoint if available
    if 'class_names' in checkpoint:
        global CLASS_NAMES
        CLASS_NAMES = checkpoint['class_names']
    
    return model


def preprocess_image(image):
    """Preprocess image for model input."""
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform (same as validation transform in training)
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def predict(image):
    """Make prediction on the input image."""
    # Preprocess image
    image_tensor = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, 3)
    
    # Format results
    results = []
    for i in range(3):
        idx = top3_indices[i].item()
        prob = top3_probs[i].item()
        results.append((CLASS_NAMES[idx], f"{prob * 100:.2f}%"))
    
    # Main prediction
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score, results


def classify_image(image):
    """Gradio interface function for image classification."""
    if image is None:
        return "Please upload an image", "No predictions available"
    
    try:
        predicted_class, confidence, top3 = predict(image)
        
        # Create result text
        result_text = f"**Predicted Class:** {predicted_class}\n"
        result_text += f"**Confidence:** {confidence:.2f}%"
        
        # Create top 3 predictions display
        top3_text = "### Top 3 Predictions:\n"
        top3_text += "\n".join([f"{i+1}. {name}: {prob}" for i, (name, prob) in enumerate(top3)])
        
        return result_text, top3_text
    except Exception as e:
        return f"Error: {str(e)}", "No predictions available"


# Load model at startup
try:
    model = load_model()
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: {str(e)}")
    model = None


# Create Gradio interface
def create_interface():
    """Create and launch Gradio interface."""
    
    with gr.Blocks(title="👕 Clothing Style Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 👕 Clothing Style Classifier
            
            Upload an image of upper-body clothing to classify it into one of 7 categories:
            
            - T-shirt
            - Polo
            - Formal Shirt
            - Tank Top
            - Sweater
            - Hoodie
            - Jacket
            
            The model uses a fine-tuned MobileNetV3-Small architecture trained on clothing images.
            """
        )
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="Upload Clothing Image",
                    type="numpy",
                    height=300
                )
                classify_btn = gr.Button("Classify", variant="primary", size="lg")
            
            with gr.Column():
                result_output = gr.Markdown(label="Prediction Result")
                top3_output = gr.Markdown(label="Top 3 Predictions")
        
        # Example images (optional - add if you have sample images)
        gr.Markdown("## 📸 Instructions")
        gr.Markdown(
            """
            1. Upload an image of upper-body clothing (cropped to show the clothing item clearly)
            2. Click the "Classify" button
            3. View the predicted class and confidence scores
            
            **Note:** For best results, use images with clear views of the clothing item, 
            similar to 224×224 pixel cropped images used during training.
            """
        )
        
        # Wire up the interface
        classify_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=[result_output, top3_output]
        )
        
        # Auto-classify on image upload
        image_input.change(
            fn=classify_image,
            inputs=image_input,
            outputs=[result_output, top3_output]
        )
    
    return demo


if __name__ == "__main__":
    if model is None:
        print("ERROR: Model not loaded. Cannot start demo.")
        print("Please ensure you have trained the model and it exists at:", MODEL_PATH)
    else:
        demo = create_interface()
        demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860
        )
