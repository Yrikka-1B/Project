# Your AI Studio Challenge Project Title Here

⚠️ _Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and examples in this template before finalizing your README._

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Mahika Acharya   | @mahikaacharya| Model training, evaluation scripts, data cleaning pipeline               |
| Jordan Ramirez   | @jramirez     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Amina Hassan     | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Priya Mehta      | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |
| Chris Park       | @chrispark    | Model evaluation, performance analysis, results interpretation           |

---

## 🎯 **Project Highlights**

- Fine-tuned a YOLO11n object detection model on synthetic data to detect 5 everyday objects (potted plant, chair, cup, vase, book)
- Improved model performance from ___ to ___ mAP@50 through data cleaning and fine-tuning
- Implemented model-guided annotation correction using CVAT to fix 10% of incorrect synthetic labels
- Developed comprehensive evaluation pipeline comparing baseline vs. fine-tuned model performance
- Demonstrated the effectiveness of synthetic data for training object detection models

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

This project is part of the **Break Through Tech AI Program** in collaboration with **YRIKKA**, a NYC-based AI startup specializing in synthetic data generation for autonomous systems.

### Objective
Evaluate whether augmenting a YOLO object detection model with synthetic data can improve real-world performance, addressing the synthetic-to-real (sim2real) data gap.

### Real-World Significance
Autonomous systems (drones, robots, vehicles) require robust perception systems, but collecting real-world training data for every possible scenario—different lighting, weather, occlusions—is prohibitively expensive. Synthetic data offers a scalable, cost-effective alternative. This project demonstrates that properly cleaned and curated synthetic data can effectively train models that generalize to real-world conditions.

### Project Scope
- Clean and correct imperfect synthetic training data (~10% incorrect annotations)
- Fine-tune YOLO11 model on cleaned synthetic dataset
- Collect 200 challenging real-world test images
- Compare baseline vs. fine-tuned model performance
- Target: 0.10 mAP@50 improvement
---

## 📊 **Data Exploration**

### Dataset Details
- **Source**: YRIKKA synthetic data engine
- **Size**: 2,000 images
- **Format**: COCO JSON annotations converted to YOLO format
- **Classes**: 5 everyday objects
  - Potted plant
  - Chair
  - Cup
  - Vase
  - Book

### Data Preprocessing

1. **Format Conversion**: Converted COCO JSON annotations to YOLO txt format
2. **Model-Guided Cleaning**: Used YOLO11 inference to identify low-confidence predictions (bottom 10%) indicating annotation errors
3. **Manual Correction**: Fixed annotations in CVAT (Computer Vision Annotation Tool)
4. **Invalid Label Removal**: Cleaned labels with class IDs outside 0-4 range (leftover from COCO's 80-class system)
5. **Train/Val Split**: 80-20 split (794 training, 199 validation images)

### Key Insights from EDA

**Class Distribution (Validation Set):**
- Chair: __ instances (__%) - Most represented
- Book: __ instances (__%)
- Cup: __ instances (__%)
- Vase: __ instances (__%)
- Potted plant: __ instances (__%) - Severely underrepresented

**Challenges:**
- Severe class imbalance affecting model performance
- Inconsistent annotation quality in original synthetic data
- Invalid class IDs from COCO-to-YOLO conversion
- Duplicate bounding boxes removed during training
  
**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

### Model Architecture
- **Base Model**: YOLO11n (nano variant)
- **Pre-trained Weights**: COCO dataset (80 classes)
- **Fine-tuning**: Specialized for 5 target classes

### Training Configuration
- **Epochs**: 10 (limited by CPU constraints; recommend 30-50 with GPU)
- **Batch Size**: 16
- **Device**: CPU (3.88 hours for 10 epochs)
- **Data Augmentation**: Horizontal flip, brightness/contrast adjustment, rotation

### Training Strategy
1. Loaded pre-trained YOLO11n weights
2. Froze early layers to retain general feature extraction
3. Fine-tuned detection head and later layers on cleaned synthetic data
4. Validated on held-out synthetic validation set
5. Monitored mAP@50, precision, and recall metrics

### Key Decisions
- Used model-guided annotation to identify errors rather than random sampling
- Applied stratified splitting to maintain class distribution (though limited by severe imbalance)
- Chose YOLO11n for balance of speed and accuracy
---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

### Limitations
- **Class imbalance**: Severely underrepresented classes need more training data
- **Real-world evaluation pending**: Synthetic validation results need confirmation on real-world test set
- **Annotation consistency**: Multiple annotators may introduce inconsistencies

### Future Work

1. **Extended Training**: Train for 30-50 epochs with GPU for full convergence
2. **Balance Dataset**: Generate additional synthetic images for underrepresented classes
3. **Domain Adaptation**: Explore techniques to reduce synthetic-to-real gap
4. **Annotation Guidelines**: Establish formal protocols for consistent labeling
5. **Class Weighting**: Implement loss weighting to handle imbalance during training
6. **Advanced Augmentation**: Add more sophisticated augmentations (cutout, mixup)
7. **Model Comparison**: Test larger YOLO variants (YOLOv8s, YOLOv8m) for performance gains
8. **Object Variety**: Test model on various objects other than our standard 5


---

## 📝 **License**

If applicable, indicate how your project can be used by others by specifying and linking to an open source license type (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type.

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

1. Ultralytics YOLO11 Documentation: https://docs.ultralytics.com/
2. CVAT (Computer Vision Annotation Tool): https://www.cvat.ai/
3. Mean Average Precision (mAP) Explained: https://blog.roboflow.com/mean-average-precision/
4. COCO Dataset: https://cocodataset.org/
5. YRIKKA Synthetic Data Platform: https://yrikka.com/
---

## 🙏 **Acknowledgements** (Optional but encouraged)
We would like to thank:

- **YRIKKA Team** (Kia Khezeli, John Kalantari, Maxim Clouser) for providing the synthetic dataset, project guidance, and domain expertise in autonomous systems
- **Break Through Tech AI Program** for the opportunity to work on this real-world AI challenge
- **AI Studio Coach** (Swagath Babu) for supporting our learning and troubleshooting

Special thanks to the Break Through Tech AI community for fostering collaboration and learning throughout this program!

