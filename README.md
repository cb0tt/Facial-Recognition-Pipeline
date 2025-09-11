# Facial Recognition Pipeline with Deep Learning

## 📌 Overview

### The pipeline...
- Detect and align faces from raw images.
- Generate **128/512-dimensional embeddings** using a pretrained **FaceNet model**.
- Train a **Support Vector Machine (SVM)** classifier on those embeddings.
- Evaluate the classifier on **unseen images** to test its ability to recognize identities.

This pipeline demonstrates the end-to-end process of building a face recognition system with **deep learning for feature extraction** and **machine learning for classification**.

## ⚙️ How It Works
1. **Face Detection & Alignment**  
   - Uses **dlib’s 68-point landmark predictor** to detect/align faces.  
   - Aligns faces by warping them to a consistent orientation and size (160×160).  
   - Ensures that eyes, nose, and mouth landmarks are in standardized positions.  
   - Output: cropped, normalized aligned face images.

2. **Face Embeddings with FaceNet**  
   - Loads a pretrained **FaceNet model (.pb)** into TensorFlow 1.x.  
   - Each aligned face is passed through the network.  
   - Output: a **512-dimensional embedding vector** representing that face’s features.

3. **Classification with SVM**  
   - Embeddings are collected and labeled per identity.  
   - A **linear SVM classifier** (from scikit-learn) is trained on these embeddings.  
   - Classes with fewer than **10 images** are dropped.  
   - The classifier is saved as a pickle file for later inference.

4. **Evaluation**  
   - New raw images are aligned using the same dlib predictor.  
   - Their embeddings are computed via FaceNet.  
   - The trained SVM predicts the identity of each embedding.  
   - Reports **overall accuracy** and **per-class accuracy** on the evaluation set.

---

## 🛠️ Tools (use Python 3.6-3.7)
- **Docker** → reproducible environment to run TensorFlow and dependencies  
- **TensorFlow** → loads the pretrained FaceNet graph and computes embeddings
- **dlib** → face detection & facial landmark alignment  
- **OpenCV (cv2)** → image preprocessing (resize, color conversion, affine warp)  
- **scikit-learn** → SVM classifier and evaluation metrics
---
## 📁 Directory Setup (Required Folders)
make sure the following directories exist in the project root:

## etc/ → Model & Classifier Storage

This folder stores:

The pretrained FaceNet model (.pb) you download

The trained SVM classifier (.pkl) after training

⚠️ These files are too large to upload to GitHub, so they must be downloaded manually.
Example model: 20180408-102900.pb
Saved classifier: facenet_svm.pkl

## data/ → All Image Data (Input/Output)

This folder is where your raw images, aligned faces, and new test images live.
You are expected to structure it like this:

data/
├── raw/               # Raw images organized in folders by person
│   └── PersonA/*.jpg
│   └── PersonB/*.jpg
├── aligned/           # Aligned output of raw images (auto-generated)
├── aligned_new/       # Aligned output of test images (auto-generated)
├── new_images/        # Ra

### 📥 Required Downloads

Before training or evaluating, download the following files and place them in the correct directories:

#### 🔹 1. FaceNet Pretrained Model (.pb)
Download:  
 📎 [20180408-102900.pb](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
📁 Place into: `etc/20180408-102900/`

### 🔹 2. dlib 68-Point Shape Predictor
- Source: [dlib model download page (SourceForge)](https://sourceforge.net/projects/dclib/files/)
- File to download: `shape_predictor_68_face_landmarks.dat.bz2`
---
### 🚀 How to Run the Code
### 1A. Clone the repository
Run below command,
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
#### 1B.  Build the Docker Image
If you haven’t already built the image locally, run:
```bash
docker build -t facenet-pipeline .
```
### 2. Preprocess (align) the raw dataset
Place your raw images into data/raw/<PersonName>/*.jpg.
Data i used: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/code 
Run this command, it creates the aligned faces in data\aligned
```bash
docker run --rm -it -v ${PWD}:/app -w /app facenet-pipeline bash -lc "python medium_facenet/preprocess.py --input_dir data/raw/lfw-deepfunneled --output_dir data/aligned --landmark_path medium_facenet/shape_predictor_68_face_landmarks.dat --image_size 160 --upsample 1"

```
### 3. Train the classifier
Run this Command, post training the classifier will be saved to etc/facenet_svm.pkl and print the list of trained classes (identities)
I filtered out classes that had < 10 images for higher confidence intervals. You can adjust however you'd like, just change 'min_per_class 10'. The more info the svm had to evaluate off of will directly impact its level of confidence in the next step.
```bash
docker run --rm -it -v ${PWD}:/app -w /app facenet-pipeline bash -lc "python medium_facenet/train_classifier.py --aligned_dir data/aligned --model_path etc/20180408-102900/20180408-102900.pb --out_pickle etc/facenet_svm.pkl --image_size 160 --batch 64 --min_per_class 10 --is_train"

```
### 4. Evaluate on new images
In data Make a folder called new_images, data/new_images, add new photos of indentities. **Ensure you use identities that the classifier has been trained on**
Preprocess command:
```bash
docker run --rm -it -v ${PWD}:/app -w /app facenet-pipeline \
bash -lc "python medium_facenet/preprocess.py \
  --input_dir data/new_images \
  --output_dir data/aligned_new \
  --landmark_path medium_facenet/shape_predictor_68_face_landmarks.dat \
  --image_size 160 --upsample 2"
```
#### Evaluate, we remove is_train to evaluate these new images without retraining
Command:
```bash
docker run --rm -it -v ${PWD}:/app -w /app facenet-pipeline \
bash -lc "python medium_facenet/train_classifier.py \
  --aligned_dir data/aligned_new \
  --model_path etc/20180408-102900/20180408-102900.pb \
  --out_pickle etc/facenet_svm.pkl \
  --image_size 160 --batch 64 \
  --min_per_class 1"
```
##### Example Output:
'Evaluated on 6 images (kept 6 that matched trained classes)
Overall accuracy: 1.0000

Sample per-class accuracy (first 10 classes):
Bill_Gates                     1.000 (2 samples)
Michael_Jackson                1.000 (2 samples)
Tom_Hanks                      1.000 (2 samples)'
### As the number and diversity of evaluation images increases, you get a more realistic and robust measure of accuracy. So if we increase the sample size...
-More variability (lighting, pose, occlusion, background).

-More opportunities for mistakes (especially between lookalike people).

-Edge cases or noisy images might confuse the model.

---
## Final Thoughts
This project successfully combined machine learning (SVM) with deep learning feature extraction (FaceNet) in a fully containerized pipeline. By aligning face images, generating embeddings, and enforcing identity thresholds during training, we created a repeatable system for facial recognition and evaluation. The use of Docker ensures that anyone can reproduce results reliably, making this pipeline practical for both experimentation and deployment.
#### 🙏 Acknowledgments
This project was originally inspired by [Cole Murray's facial recognition pipeline tutorial](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8). While the core ideas and flow were based on his tutorial, this implementation makes several key changes — including code refactoring, updated preprocessing, classifier handling, and a fully customized execution pipeline designed for learning and reproducibility.


