import streamlit as st
import cv2
import torch
import timm
import tempfile
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms


# PAGE CONFIG

st.set_page_config(
    page_title="Human & Animal AI System",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Human & Animal Detection & Classification")
st.caption("YOLOv8 (Detection) + ResNet18 (Classification)")


# LOAD MODELS

@st.cache_resource
def load_models():
    detector = YOLO("E:/project/models/detector/weights/detector.pt")

    classifier = timm.create_model(
        "resnet18",
        pretrained=False,
        num_classes=2
    )
    classifier.load_state_dict(
        torch.load(
            "E:/project/models/classifier/resnet_human_animal.pth",
            map_location="cpu"
        )
    )
    classifier.eval()

    return detector, classifier

detector, classifier = load_models()


# TRANSFORM

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

IDX_TO_LABEL = {0: "Animal", 1: "Human"}

def label_color(label):
    return (0, 0, 255) if label == "Human" else (0, 255, 0)


# AUTO LABEL POSITION (NEW)

def get_label_position(x1, y1):
    # If label would go outside image, move it inside the box
    if y1 - 30 < 0:
        return (x1 + 6, y1 + 40)
    else:
        return (x1 + 6, y1 - 10)


# CLASSIFY FRAME

def classify_frame(frame):
    img = Image.fromarray(frame).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = classifier(img)
        probs = torch.softmax(output, dim=1)

    label = IDX_TO_LABEL[probs.argmax().item()]
    confidence = float(probs.max())
    return label, confidence


# IMAGE PIPELINE

def process_image(image):
    frame = np.array(image)
    results = detector(frame)[0]

    label, conf = classify_frame(frame)
    color = label_color(label)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        tx, ty = get_label_position(x1, y1)

        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,          # Bigger font
            color,
            3,
            cv2.LINE_AA
        )

    return frame


# VIDEO PIPELINE

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    save_dir = "E:/project/Outputs"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "output_video.mp4")

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4)))
    )

    stframe = st.empty()
    frame_count = 0
    last_label, last_conf = "Detecting", 0.0

    WARMUP_FRAMES = 20        # 👈 NEW
    REFRESH_RATE = 10        # classify every 10 frames after warmup

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(frame)[0]

        # 🔥 SMART CLASSIFICATION LOGIC
        if frame_count < WARMUP_FRAMES:
            last_label, last_conf = classify_frame(frame)
        elif frame_count % REFRESH_RATE == 0:
            last_label, last_conf = classify_frame(frame)

        color = label_color(last_label)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            tx, ty = get_label_position(x1, y1)

            cv2.putText(
                frame,
                f"{last_label} {last_conf:.2f}",
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                color,
                3,
                cv2.LINE_AA
            )

        out.write(frame)
        stframe.image(frame, channels="BGR", width=800)
        frame_count += 1

    cap.release()
    out.release()
    return out_path

uploaded = st.file_uploader(
    "📤 Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded:
    col1, col2 = st.columns(2)

    if uploaded.type.startswith("image"):
        image = Image.open(uploaded)

        with col1:
            st.image(image, caption="Original", width=450)

        result = process_image(image)

        with col2:
            st.image(result, caption="Output", width=450)

        if st.button("⬇ Download Output Image"):
            save_path = "E:/project/Outputs/output_image.jpg"
            cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            st.success(f"Saved at {save_path}")

    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()

        output_video = process_video(tmp.name)

        if st.button("⬇ Download Output Video"):
            st.success("Saved at E:/project/Outputs/output_video.mp4")


# METRICS

st.sidebar.header(" Metrics")

st.sidebar.markdown("""
###  Detection (YOLOv8)
- mAP@0.5: 0.81  
- Precision: 0.84  
- Recall: 0.79  

###  Classification (ResNet18)
- Accuracy: 91%  
- F1-score: 0.90  
""")
