from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os

# Definir clase del modelo (igual que en el notebook)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
# Configurar Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo
model = CNN()
model.load_state_dict(torch.load('modelo_cnn.pt', map_location=torch.device('cpu')))
model.eval()

# Transformación
transform = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(),             
])

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0)

            output = model(image)
            _, predicted = torch.max(output, 1)
            classes = ['Gato', 'Perro']
            prediction = classes[predicted[0].item()]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
