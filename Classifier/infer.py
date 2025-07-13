import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import EnhancedCNN

# Load class names from the text file
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to perform inference on an image
def predict_image(image_path, model_path, class_names, threshold=0.5):
    # Instantiate the model and load the trained weights
    model = EnhancedCNN(num_classes=len(class_names))
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            weights_only=True
        )
    )
    model.eval()  # Set the model to evaluation mode

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Match input size during training
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        outputs = model(image)  # Forward pass
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert to probabilities
        confidence, predicted = torch.max(probabilities, 1)  # Get the max probability and corresponding class index
        predicted_class = predicted.item()
        predicted_label = class_names[predicted_class]
    
    # Check confidence level
    if confidence.item() < threshold:
        predicted_label = "Unforeseen Image"

    return predicted_class, predicted_label, confidence.item()

# Predict on a sample image
image_path = 'C:\Users\23100036\Desktop\na0281\Classifier\images'
model_path = 'C:\\Users\\23100036\\Desktop\\na0281\\Classifier\\enhanced_cnn_best.pth'  # Updated model path to match saved model
predicted_class, predicted_label, confidence = predict_image(image_path, model_path, class_names)

# Print the predicted class, label, and confidence
print(f'The predicted class is: {predicted_class} (Label: {predicted_label}), Confidence: {confidence:.2f}')

# Visualize the image with the prediction
def imshow(image_path, predicted_label):
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()

# Display the image
imshow(image_path, predicted_label)
