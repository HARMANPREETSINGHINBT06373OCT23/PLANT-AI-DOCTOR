from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
import torch.optim as optim

# ========== Flask Setup ==========
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ========== CNN Model ==========
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ========== Classes & Transform ==========
classes = [
    "Tomato_Bacterial_Spot", "Tomato_Early_Blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_Spot", "Tomato_Yellow_Leaf_Curl", "Tomato_Healthy",
    "Potato_Early_Blight", "Potato_Late_Blight", "Potato_Healthy",
    "Corn_Common_Rust"
]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ========== Fake Dataset ==========
dataset = FakeData(size=100, image_size=(3, 64, 64), num_classes=len(classes), transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ========== Train Tiny CNN ==========
device = torch.device("cpu")
model = TinyCNN(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):  # short training
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()

# ========== Disease Info Dictionary ==========
disease_info = {
    "Tomato_Bacterial_Spot": {"definition": "Bacterial spots on leaves and fruits.", "color": "red", "health_status": "Unhealthy"},
    "Tomato_Early_Blight": {"definition": "Fungal dark spots on older leaves.", "color": "brown", "health_status": "Unhealthy"},
    "Tomato_Leaf_Mold": {"definition": "Yellow spots with mold under leaves.", "color": "orange", "health_status": "Unhealthy"},
    "Tomato_Septoria_Spot": {"definition": "Gray-centered circular spots.", "color": "gray", "health_status": "Unhealthy"},
    "Tomato_Yellow_Leaf_Curl": {"definition": "Curling and yellowing of leaves.", "color": "yellow", "health_status": "Unhealthy"},
    "Tomato_Healthy": {"definition": "Healthy tomato leaf.", "color": "green", "health_status": "Healthy"},
    "Potato_Early_Blight": {"definition": "Brown spots with rings on leaves.", "color": "brown", "health_status": "Unhealthy"},
    "Potato_Late_Blight": {"definition": "Rapid lesions on leaves and stems.", "color": "darkred", "health_status": "Unhealthy"},
    "Potato_Healthy": {"definition": "Healthy potato foliage.", "color": "green", "health_status": "Healthy"},
    "Corn_Common_Rust": {"definition": "Red-brown pustules on corn leaves.", "color": "red", "health_status": "Unhealthy"}
}

# ========== Routes ==========
@app.route('/')
def index():
    return render_template('index.html')  # ✅ Connects Flask to your HTML UI

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = classes[predicted.item()]

    info = disease_info.get(predicted_class, {
        'definition': 'Unknown disease.',
        'color': 'gray',
        'health_status': 'Unknown'
    })

    return jsonify({
        'result': predicted_class,
        'definition': info['definition'],
        'color': info['color'],
        'healthy': info['health_status'] == "Healthy",
        'image_path': filepath
    })

# ========== Run App ==========
if __name__ == '__main__':
    app.run(debug=True)
