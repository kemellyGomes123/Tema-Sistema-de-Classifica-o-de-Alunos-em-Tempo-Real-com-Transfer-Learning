from flask import Flask, render_template, request, Response
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import cv2

app = Flask(__name__)

# Carregar o modelo treinado com pesos atualizados
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Ajuste o caminho conforme necessário
model.load_state_dict(torch.load('C:\\Users\\kemelly Gomes\\OneDrive\\Desktop\\Projecto.final_IA\\model.pth'))
model.eval()

# Transformações
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        img = Image.open(io.BytesIO(file.read()))
        img = data_transforms(img).unsqueeze(0)
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        class_names = ['pessoas', 'jimin']
        result = class_names[preds[0]]
        return result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = data_transforms(img).unsqueeze(0)
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            class_names = ['pessoas', 'jimin']
            label = class_names[preds[0]]
            
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

if __name__ == '__main__':
    app.run(debug=True)
