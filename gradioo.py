import torch
import requests
import gradio as gr

from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


# Load model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()


# Download ImageNet labels
response = requests.get("https://git.io/JJkYN")
labels = [l.strip() for l in response.text.split("\n") if l.strip()]


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def predict(inp):
    inp = transform(inp).unsqueeze(0)

    with torch.no_grad():
    prediction = torch.nn.functional.softmax(
            model(inp)[0], dim=0
        )

    confidences = {
        labels[i]: float(prediction[i])
        for i in range(len(labels))
    }

    return confidences


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    examples=["/content/lion.jpg", "/content/cheetah.jpg"]
).launch()