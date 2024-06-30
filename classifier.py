from torchvision import models, transforms
from PIL import Image
import torch

def load_model(architecture):
    if architecture == "resnet":
        model = models.resnet50(pretrained=True)
    elif architecture == "alexnet":
        model = models.alexnet(pretrained=True)
    elif architecture == "vgg":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError("Invalid model architecture")
    
    model.eval()
    return model

def classify_image(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
    
    _, predicted = outputs.max(1)
    return predicted.item()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
