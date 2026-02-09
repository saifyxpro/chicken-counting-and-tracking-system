# 🐔 Chicken Counting & Tracking System

Real-time chicken counting and tracking using **YOLOv11s** + **ByteTrack** with Gradio UI.

![Demo](assets/demo.png)

## ✨ Features

- **YOLOv11s Detection** - Fine-tuned for chicken detection
- **Sequential Counting** - Chickens numbered #1, #2, #3...
- **Video Tracking** - ByteTrack for unique ID assignment
- **CPU Inference** - Runs on CPU by default
- **Gradio Web UI** - Easy image/video upload

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/yourusername/chicken-counting.git
cd chicken-counting

# Install
pip install -e .

# Run
python app.py
```

Open: http://localhost:7860

## 📁 Project Structure

```
chicken-counting/
├── app.py              # Gradio web app
├── pyproject.toml      # Dependencies
├── src/
│   ├── detector.py     # YOLOv11s detection
│   └── tracker.py      # ByteTrack tracking
├── models/
│   └── chicken_yolov11s.pt  # Trained model
├── assets/
│   └── demo.png        # Demo image
└── outputs/            # Results
```

## 🧠 Model

| Model                   | Description                               |
| ----------------------- | ----------------------------------------- |
| **chicken_yolov11s.pt** | YOLOv11s fine-tuned for chicken detection |

## 📊 Datasets

| Dataset                      | Images | Link                                                                                             |
| ---------------------------- | ------ | ------------------------------------------------------------------------------------------------ |
| Chicken Detection & Tracking | 463    | [Roboflow](https://universe.roboflow.com/chickens/chicken-detection-and-tracking)                |
| Chicken YOLO                 | 38     | [Roboflow](https://universe.roboflow.com/od-3hysf/chicken-yolo)                                  |
| Kaggle Chicken               | 106+   | [Kaggle](https://www.kaggle.com/datasets/nirmalsankalana/chicken-detection-and-tracking-dataset) |

## 👤 Author

**Saifullah Channa** - [hello@saify.me](mailto:hello@saify.me)

## 📄 License

Apache 2.0
