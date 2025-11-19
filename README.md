# 🚨 Smart Traffic Anomaly Detection System

## 📖 What is This Project?

This project uses **Artificial Intelligence** to watch traffic videos and automatically detect **unusual events** like accidents, congestion, and traffic anomalies. Instead of manually watching hours of video footage, this system analyzes every frame and creates a detailed report showing exactly **when** and **where** problems occurred.

### Real-World Use Case:

Imagine a traffic management center in a city with 100 surveillance cameras. Instead of having people watch each camera 24/7, this system watches all cameras automatically, detects problems in seconds, and alerts operators with timestamps and severity levels.

---

## 🎯 What Does It Do? (Simple Explanation)

```
INPUT: Traffic video file (e.g., highway, intersection)
   ↓
PROCESS: AI analyzes each frame to detect patterns
   ↓
OUTPUT: JSON report with:
   - When anomalies occurred (timestamp)
   - How severe they are (anomaly score)
   - Traffic congestion level (HIGH/MEDIUM/LOW)
   - Potential accident locations
```

---

## 🔄 How It Works (Step-by-Step)

### Step 1: Extract All Frames

```
Video: 14,000 frames for 254 Videos (2 minutes at 30 FPS)
   ↓
System extracts every single frame as an image
   ↓
Result: 14,000 individual JPG images saved
```

**Why?** Videos are too large to process directly. We break them into frames.

---

### Step 2: Smart Sampling (Quantization)

```
14,000 frames extracted
   ↓
Take 1 frame every 3 frames
   ↓
Result: ~4,667 frames (33% of original)
```

**Why?** We keep enough frames to detect patterns but reduce processing time.

---

### Step 3: Feature Extraction Using AI

```
Frame image
   ↓
Vision Transformer (ViT) model analyzes it
   ↓
Creates a "fingerprint" (768 numbers) representing what the model sees
   ↓
Result: Feature vector stored for analysis
```

**Why?** Raw pixels are meaningless. The AI converts images into meaningful numbers it can understand.

---

### Step 4: Anomaly Detection

```
4,667 feature vectors
   ↓
Isolation Forest model learns what "normal traffic" looks like
   ↓
Scores each frame: Lower score = more unusual
   ↓
Result: Anomaly score for each frame (-0.85 to +0.92)
```

**Why?** The model identifies frames that don't match normal patterns (accidents, congestion, incidents).

---

### Step 5: Classification

```
For each frame, the system determines:
   ✓ Timestamp (MM:SS)
   ✓ Anomaly score (numerical)
   ✓ Congestion level (HIGH/MEDIUM/LOW)
   ✓ Is it a potential accident? (YES/NO)
```

---

### Step 6: Report Generation

```
All analyzed data → Organized into JSON format
   ↓
Includes summary statistics
   ↓
Ready for download and analysis
```

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────┐
│   Your Traffic Video File       │
│   (e.g., highway.avi)           │
└────────────┬────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ PHASE 1: EXTRACT   │
    │ ALL FRAMES         │
    │ No skipping        │
    └────────┬───────────┘
             │
             ▼
    14,000 raw frames
    (complete video data)
             │
             ▼
    ┌────────────────────┐
    │ PHASE 2: QUANTIZE  │
    │ (Smart sampling)   │
    │ Take 1/3 frames    │
    └────────┬───────────┘
             │
             ▼
    4,667 quantized frames
    (33% of original)
             │
             ▼
    ┌────────────────────────┐
    │ PHASE 3: ViT FEATURES  │
    │ AI extracts patterns   │
    │ 768-dim vectors        │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────────┐
    │ PHASE 4: ANOMALY DETECTION │
    │ Isolation Forest model     │
    │ Scores each frame          │
    └────────┬───────────────────┘
             │
             ▼
    ┌────────────────────────────┐
    │ PHASE 5: CLASSIFICATION    │
    │ - Timestamps               │
    │ - Congestion levels        │
    │ - Accident detection       │
    └────────┬───────────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ PHASE 6: JSON REPORT   │
    │ Ready for download     │
    └────────────────────────┘
```

---

## 📁 Project Structure

```
smart-traffic-anomaly-detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── frames_raw/                        # Step 1 output
│   └── traffic_video_1/
│       ├── frame_000001.jpg
│       ├── frame_000002.jpg
│       └── ... (14,000 frames)
│
├── frames_quantized/                  # Step 2 output
│   └── traffic_video_1/
│       ├── frame_00001.jpg
│       ├── frame_00002.jpg
│       └── ... (~4,667 frames)
│
└── results/                           # Step 6 output
    ├── traffic_anomaly_detection_report.json
    ├── 01_raw_frames_distribution.png
    ├── 02_raw_vs_quantized_comparison.png
    └── 03_sample_frames.png
```

---

## 🚀 How to Run (Google Colab)

### Prerequisites:

* Google account
* Traffic video file (MP4, AVI, or MOV format)
* 2-3 hours of GPU time (free from Colab)

### Step-by-Step:

1. **Open Google Colab**
   ```
   Go to: colab.research.google.com
   Create new notebook
   ```
2. **Upload Your Video**
   ```
   Click: Files → Upload → Select your video
   Wait for upload to complete
   ```
3. **Run Cells 1-8 (Extract & Quantize)**
   ```
   Copy Cell 1 → Run
   Copy Cell 2 → Run
   ... (continue for all 8 cells)
   Expected time: 5-10 minutes
   ```
4. **Run Cells 9-18 (Analysis)**
   ```
   Copy Cell 9 → Run
   ... (continue through Cell 18)
   Expected time: 20-30 minutes
   The JSON file will download automatically!
   ```

---

## 📊 Understanding the Output (JSON Report)

### Sample Report Structure:

```json
{
  "metadata": {
    "model_name": "vit_base_patch16_224",
    "anomaly_detector": "IsolationForest"
  },
  "summary": {
    "total_frames_analyzed": 4667,
    "anomalies_detected": 233,
    "potential_accidents": 23,
    "congestion_level_counts": {
      "HIGH": 1555,
      "MEDIUM": 1556,
      "LOW": 1556
    }
  },
  "frames_data": [
    {
      "frame_id": "frame_00001",
      "timestamp": "00:10.50",
      "anomaly_score": -0.4523,
      "is_anomaly": true,
      "congestion_level": "HIGH",
      "is_accident": false
    }
  ]
}
```

### Key Fields Explained:


| Field                 | Meaning                | Example                              |
| --------------------- | ---------------------- | ------------------------------------ |
| **timestamp**         | When in video (MM:SS)  | "00:10.50" = 10.5 seconds            |
| **anomaly\_score**    | How unusual (-1 to +1) | -0.45 = very unusual, +0.92 = normal |
| **is\_anomaly**       | Is frame abnormal?     | true = yes, false = no               |
| **congestion\_level** | Traffic density        | HIGH/MEDIUM/LOW                      |
| **is\_accident**      | Potential accident?    | true = bottom 1% scores              |

---

## 🎓 Understanding the Key Concepts

### What is "Anomaly"?

An **anomaly** is anything unusual in the video that differs from normal traffic patterns:

* 🚗 Stalled vehicles
* 💥 Accidents or collisions
* 🚦 Unusual congestion patterns
* 🚧 Road blockages
* 👤 People on roadway

### What is "Congestion Level"?

Traffic density classification:

* **HIGH**: Heavy anomalies (unusual traffic patterns detected)
* **MEDIUM**: Moderate anomalies (some unusual activity)
* **LOW**: Normal traffic (routine patterns)

### What is "Anomaly Score"?

A number showing how unusual a frame is:

```
-0.85 ← Very unusual (likely anomaly)
-0.50 ← Unusual
 0.00 ← Neutral
+0.50 ← Normal
+0.92 ← Very normal
```

---

## 🔧 Technical Details (For Advanced Users)

### Models Used:

**Vision Transformer (ViT):**

* Pre-trained on ImageNet
* Extracts 768-dimensional features
* Better at understanding context than CNNs

**Isolation Forest:**

* Unsupervised anomaly detection
* Identifies outliers in high dimensions
* Contamination rate: 5% (expects 5% anomalies)

### Data Processing:

```
14,000 raw frames
   → Resize to 224×224 pixels
   → Normalize using ImageNet standards
   → Quantize to 4,667 frames
   → Extract ViT features
   → Normalize with StandardScaler
   → Train Isolation Forest
   → Score each frame
   → Classify by percentiles
```

---

## 📈 Performance Metrics


| Metric                 | Value                    |
| ---------------------- | ------------------------ |
| **Frames Analyzed**    | 4,667                    |
| **Processing Time**    | 25-30 minutes            |
| **GPU Memory**         | \~4-6 GB                 |
| **Output File Size**   | \~2-3 MB                 |
| **Anomalies Detected** | \~233 (5%)               |
| **Accuracy**           | Depends on training data |

---

## ⚙️ Customization Options

### Change Quantization Ratio:

```python
# In Cell 3
FRAME_SKIP_FOR_QUANTIZATION = 3  # Current: 1/3 frames

# To get more frames:
FRAME_SKIP_FOR_QUANTIZATION = 2  # Gets ~50% of frames
FRAME_SKIP_FOR_QUANTIZATION = 1  # Gets all frames

# To get fewer frames:
FRAME_SKIP_FOR_QUANTIZATION = 4  # Gets ~25% of frames
```

### Change Anomaly Sensitivity:

```python
# In Cell 12
isolation_forest = IsolationForest(
    contamination=0.05  # Current: 5% anomalies
    # Change to 0.10 for 10% anomalies (more sensitive)
    # Change to 0.02 for 2% anomalies (less sensitive)
)
```

---

## 🐛 Troubleshooting

### Problem: "Out of Memory" Error

**Solution:**

* Use higher quantization (FRAME\_SKIP\_FOR\_QUANTIZATION = 4)
* Process shorter videos
* Use free GPU in Colab

### Problem: Very Few Anomalies Detected

**Solution:**

* Lower contamination rate (0.02 instead of 0.05)
* Check if video has actual anomalies
* Verify quantization is working

### Problem: Timestamps Look Wrong

**Solution:**

* Verify video FPS is detected correctly
* Check FRAME\_SKIP\_FOR\_QUANTIZATION value
* Frame index × (skip\_interval / FPS) = timestamp

---

## 📚 References & Resources

### Papers:

* Vision Transformer (ViT): https://arxiv.org/abs/2010.11929
* Isolation Forest: https://cs.anu.edu.au/wp-content/uploads/2015/06/Isolation-Forest.pdf

### Libraries Used:

* PyTorch: https://pytorch.org/
* Timm (PyTorch Image Models): https://github.com/rwightman/pytorch-image-models
* Scikit-learn: https://scikit-learn.org/

---

## 💡 Real-World Applications

1. **Traffic Management Centers**
   * Monitor multiple cameras
   * Alert operators to incidents
   * Reduce response time
2. **Highway Monitoring**
   * Detect accidents in real-time
   * Track congestion patterns
   * Optimize traffic flow
3. **Parking Lot Surveillance**
   * Detect unusual behavior
   * Monitor traffic density
4. **City Planning**
   * Analyze traffic patterns
   * Identify problem areas
   * Plan infrastructure improvements

---

## 👨‍💻 Author & Attribution

**Project Name:** Smart Traffic Anomaly Detection System

**Technology Stack:**

* Vision Transformer (Meta AI)
* Isolation Forest (Scikit-learn)
* PyTorch Deep Learning Framework

---

## 📞 Support & Questions

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all dependencies are installed (requirements.txt)
3. Ensure video file is valid and not corrupted
4. Check GPU memory availability in Colab

---

## 📝 Project Summary


| Aspect              | Details                                         |
| ------------------- | ----------------------------------------------- |
| **Purpose**         | Detect traffic anomalies in surveillance videos |
| **Input**           | Video file (AVI, MP4, MOV)                      |
| **Output**          | JSON report with timestamps & classifications   |
| **Processing Time** | \~30 minutes for 14k frames                     |
| **Accuracy**        | Depends on training data & model tuning         |
| **Use Case**        | Traffic management, incident detection          |
| **Scalability**     | Can handle multiple videos sequentially         |

---

## 🎉 Conclusion

This project demonstrates how **Artificial Intelligence** can automate surveillance analysis, making traffic management more efficient and responsive. By using advanced deep learning models and anomaly detection techniques, we can identify unusual events automatically, saving time and improving public safety.

**Key Takeaway:** Instead of humans watching videos 24/7, AI does the watching and alerts humans only when something unusual happens.

# AIML-Project
