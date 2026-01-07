# Facial Recognition Pipeline (FaceNet + SVM)

This repo aligns faces with dlib, extracts embeddings with TensorFlow, trains a linear SVM classifier, and predicts identities for new images.

## Requirements

- Docker Desktop (Linux containers / WSL2 engine)
- A FaceNet `.pb` graph file

## Expected folders

- `data/raw/lfw-deepfunneled/` (or your own training dataset, organized as `PersonName/*.jpg`)
- `data/aligned/` (generated)
- `data/new_images/` (your “new” raw images, organized as `PersonName/*.jpg` or any nesting you want)
- `data/aligned_new/` (generated)

## Run (Docker)

### 1) Build image

```bash
docker build -t facenet-pipeline .
```

### 2) Align training images → `data/aligned`

```bash
docker run --rm -it -v ${PWD}:/app -w /app facenet-pipeline bash -lc "python medium_facenet/preprocess.py --input_dir data/raw/lfw-deepfunneled --output_dir data/aligned --landmark_path medium_facenet/shape_predictor_68_face_landmarks.dat --image_size 160 --upsample 1"
```

### 3) Train SVM classifier → `etc/facenet_svm.pkl`

```bash
docker run --rm -it -v ${PWD}:/app -w /app facenet-pipeline bash -lc "python medium_facenet/train_classifier.py --aligned_dir data/aligned --model_path etc/20180408-102900/20180408-102900.pb --out_pickle etc/facenet_svm.pkl --image_size 160 --batch 64 --min_per_class 10 --is_train"
```

### 4) Align new images → `data/aligned_new`

```bash
docker run --rm -it -v ${PWD}:/app -w /app facenet-pipeline bash -lc "python medium_facenet/preprocess.py --input_dir data/new_images --output_dir data/aligned_new --landmark_path medium_facenet/shape_predictor_68_face_landmarks.dat --image_size 160 --upsample 2"
```

### 5) Predict on new aligned images (prints + CSV)

```bash
docker run --rm -it -v ${PWD}:/app -w /app facenet-pipeline bash -lc "python medium_facenet/train_classifier.py --aligned_dir data/aligned_new --model_path etc/20180408-102900/20180408-102900.pb --out_pickle etc/facenet_svm.pkl --image_size 160 --batch 64 --top_k 3 --out_csv predictions.csv"
```
