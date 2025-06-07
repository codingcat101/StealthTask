# ⚽ Soccer Player & Ball Tracker using YOLOv11 + ByteTrack

This project performs automated detection and tracking of **soccer players**, **referees**, and the **ball** in a video using **YOLOv11** [Object detection model link: https://drive.google.com/file/d/1-5fOSHOSB9UXyP enOoZNAMScrePVcMD/view
Note: The model is a basic fine-tuned version of Ultralytics YOLOv11, trained for players and the ball.] and **ByteTrack** via the [Supervision](https://github.com/roboflow/supervision) library. The result is an annotated video where players and referees are marked with ellipses and the ball with a triangle.

---

## 📌 Features

- Custom YOLOv11 model loading
- ByteTrack-based object ID tracking across frames
- Ball, player, referee distinction with custom shapes
- Intermediate `.pkl` caching to speed up re-runs
- Read and write video with OpenCV

---

## 🛠️ Setup Instructions (Google Colab)

1. **Install dependencies:**
```python
!pip install ultralytics supervision opencv-python-headless
```

2. **Mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Place these files in Google Drive:**
   - `soccer.mp4` (input video)
   - `best.pt` (YOLOv11 trained model)
   - Optional: `player_detection.pkl` (saved detection info)

4. **Run your pipeline using:**
```python
!python main.py
```

---

## 📂 File Structure

```
📦project/
 ┣ 📄main.py               # Main runner script
 ┣ 📄tracker.py            # Tracker class: detection, tracking, visualization
 ┣ 📄utils.py              # Helper functions for I/O and calculations
 ┣ 📁output_videos/        # Final annotated video output
 ┃ ┗ 📄output.avi
 
```

---

## 📋 Dependencies

| Library        | Use                          |
|----------------|-------------------------------|
| `ultralytics`  | YOLOv11 model + prediction     |
| `supervision`  | ByteTrack wrapper             |
| `opencv-python`| Read/write videos and drawing |
| `numpy`        | Math and arrays               |
| `pickle`       | Save/load intermediate results|

Install in Colab via:
```bash
!pip install ultralytics supervision opencv-python-headless
```

---

## ▶️ How to Use

- To **run everything**, simply execute:
```python
python main.py
```
- This will:
  - Read video from Google Drive
  - Run detection and tracking
  - Save output to `output_videos/output.avi`

---

## 🧠 Tracker Logic

- YOLOv11 predicts frames in batches (size = 20)
- Class `"goalkeeper"` is converted to `"player"`
- Supervision ByteTrack assigns persistent IDs
- Shapes drawn:
  - 🔴 **Ellipse** = Player/Referee
  - 🟢 **Triangle** = Ball

---

## 💡 Tips

- Set `read_from_stub=True` in `get_object_tracks()` to avoid reprocessing video.
- All files read from `/content/drive/MyDrive/` in Google Colab.

---

## 👤 Author

**Aarchi Kothari**  
Project developed as part of a sports video analysis task using CV/ML techniques.
