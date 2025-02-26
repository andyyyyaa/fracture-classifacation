from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载判断是否有骨折的模型
fracture_detection_model = models.resnet50()  # 假设是ResNet50
fracture_detection_model.fc = nn.Linear(fracture_detection_model.fc.in_features, 2)  # 假设有两个类别：有骨折，无骨折
fracture_detection_model.load_state_dict(torch.load('model2.pth', map_location=device))
fracture_detection_model = fracture_detection_model.to(device)
fracture_detection_model.eval()

# 加载骨折类型分类模型
fracture_classification_model = models.resnet50()  # 不传递任何参数
fracture_classification_model.fc = nn.Linear(fracture_classification_model.fc.in_features, 10)  # 假设有10个类别
fracture_classification_model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
fracture_classification_model = fracture_classification_model.to(device)
fracture_classification_model.eval()

# 选择设备
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 根据你的模型输入大小调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

# 定义类别标签
class_labels = ['Avulsion Fracture 撕脱性骨折', 'Comminuted Fracture 粉碎性骨折', 'Fracture Dislocation 骨折脱位', 'Greenstick Fracture 青枝骨折', 'Hairline Fracture 线性骨折', 'Impacted Fracture 嵌入性骨折', 'Longitudinal Fracture 纵向骨折', 'Oblique Fracture 斜向骨折', 'Pathological Fracture 病理性骨折', 'Spiral Fracture 螺旋骨折']  

fracture_treatment = {
    "Avulsion Fracture 撕脱性骨折": "立即停止活动，避免用力拉扯受伤部位。用冰敷减少肿胀，尽量保持骨折部位静止并前往医院治疗。",
    "Comminuted Fracture 粉碎性骨折": "由于骨折呈多片状，需避免任何形式的移动。用冷敷减轻肿胀，并使用绷带固定骨折部位，尽快就医。",
    "Fracture Dislocation 骨折脱位": "避免试图自行复位，避免对脱位部位施加压力或拉扯。使用夹板或绷带固定，尽快就医处理，通常需要手术复位。",
    "Greenstick Fracture 青枝骨折": "通常发生在儿童身上，骨折部位容易弯曲而不完全断裂。通过支撑和固定骨折部位来减少疼痛，适时就医，可能需要石膏或夹板。",
    "Hairline Fracture 线性骨折": "这种骨折通常症状较轻，但仍需休息，避免对受伤部位施加压力或负重。就医确诊后可能需要佩戴夹板或石膏。",
    "Impacted Fracture 嵌入性骨折": "骨折的两端嵌入彼此，避免强行活动或承重。用冰敷减少肿胀，避免移动并立即就医治疗，通常需要放射检查来确定是否需要手术。",
    "Longitudinal Fracture 纵向骨折": "这类骨折通常沿骨的长度方向发生。立即避免移动受伤部位，使用夹板或绷带固定，并尽快就医处理。",
    "Oblique Fracture 斜向骨折": "这种骨折通常是斜向断裂。立即避免移动骨折部位，使用夹板或支架固定，并及时就医。",
    "Pathological Fracture 病理性骨折": "此类骨折是由于原有的疾病（如肿瘤或骨质疏松）导致的，通常较为脆弱。避免施加压力，减少活动并尽快就医进行详细检查和治疗。",
    "Spiral Fracture 螺旋骨折": "通常由于扭转或拉伸的暴力引起，需避免任何形式的负重或用力，立即进行固定，并前往医院治疗，通常需要影像学检查来评估伤势。"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 读取图像并进行预处理
    image = Image.open(io.BytesIO(file.read()))
    input_tensor = preprocess(image).unsqueeze(0)  # 添加批次维度
    input_tensor = input_tensor.to(device)

    # 使用第一个模型判断是否有骨折
    with torch.no_grad():
        output = fracture_detection_model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        has_fracture = torch.argmax(probabilities, dim=1).item() == 0  # 假设1表示有骨折

    if not has_fracture:
        # 如果没有骨折，直接返回结果
        result = {
            'classification': '无骨折',
            'confidence': f'{probabilities[0][1].item() * 100:.2f}%',
            'advice': '没有检测到骨折，请保持健康。',
        }
    else:
        # 如果有骨折，使用第二个模型进行详细分类
        with torch.no_grad():
            output = fracture_classification_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            predicted_class = class_labels[predicted_class_idx.item()]
            confidence_percentage = confidence.item() * 100

        # 返回详细分类结果
        result = {
            'classification': predicted_class,
            'confidence': f'{confidence_percentage:.2f}%',
            'advice': fracture_treatment[predicted_class],
        }

    return jsonify(result)

@app.route('/result')
def result():
    classification = request.args.get('classification', '未知')
    confidence = request.args.get('confidence', '未知')
    advice = request.args.get('advice', '无建议')
    return render_template('result.html', classification=classification, confidence=confidence, advice=advice)

if __name__ == '__main__':
    app.run(debug=True，threaded=True) 
