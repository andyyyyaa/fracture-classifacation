import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
fracture_detection_model = models.resnet50()
fracture_detection_model.fc = nn.Linear(fracture_detection_model.fc.in_features, 2)

# 确保路径正确
try:
    fracture_detection_model.load_state_dict(torch.load('model2.pth', map_location=device, weights_only=False))
except Exception as e:
    st.error(f"加载 fracture_detection_model 时出错: {e}")

fracture_detection_model = fracture_detection_model.to(device)
fracture_detection_model.eval()

fracture_classification_model = models.resnet50()
fracture_classification_model.fc = nn.Linear(fracture_classification_model.fc.in_features, 10)

# 确保路径正确
try:
    fracture_classification_model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=False))
except Exception as e:
    st.error(f"加载 fracture_classification_model 时出错: {e}")

fracture_classification_model = fracture_classification_model.to(device)
fracture_classification_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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

# 读取 CSS 文件
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 读取 JavaScript 文件
def load_js(file_name):
    with open(file_name) as f:
        st.markdown(f'<script>{f.read()}</script>', unsafe_allow_html=True)

def main():
    st.title("骨折检测与分类")

    # 加载自定义 CSS 和 JavaScript
    load_css("static/styles.css")
    load_js("static/script.js")

    uploaded_file = st.file_uploader("上传一张X光片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # 确保图像是 RGB 模式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption='上传的图像', use_container_width=True)
        
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = fracture_detection_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            has_fracture = torch.argmax(probabilities, dim=1).item() == 0

        if not has_fracture:
            st.write("无骨折")
            st.write(f"置信度: {probabilities[0][1].item() * 100:.2f}%")
            st.write("建议: 没有检测到骨折，请保持健康。")
        else:
            with torch.no_grad():
                output = fracture_classification_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_class_idx = torch.max(probabilities, 1)
                predicted_class = class_labels[predicted_class_idx.item()]
                confidence_percentage = confidence.item() * 100

            st.write(f"分类: {predicted_class}")
            st.write(f"置信度: {confidence_percentage:.2f}%")
            st.write(f"建议: {fracture_treatment[predicted_class]}")

if __name__ == "__main__":
    main() 

