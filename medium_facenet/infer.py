import os
import argparse
import pickle
import numpy as np
import cv2
import tensorflow as tf  # TF 1.x in your Docker
from typing import Optional, Tuple, List

from align_dlib import AlignDlib  # your class

# ---------- FaceNet loader (same idea as in train_classifier.py) ----------
def load_facenet(pb_path: str):
    """
    Load a frozen FaceNet graph (.pb) into a TF1.x Graph + Session,
    and return the input/embedding/phase tensors you need for inference.
    """
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session(graph=g)
        with tf.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
        # These names are common; if you get a KeyError, print the graph ops and adjust.
        input_t = g.get_tensor_by_name("input:0")
        emb_t   = g.get_tensor_by_name("embeddings:0")
        phase_t = g.get_tensor_by_name("phase_train:0")
    return g, sess, {"input": input_t, "emb": emb_t, "phase": phase_t}

# ---------- Minimal image prep helpers ----------
def bgr_to_facenet_rgb(img_bgr: np.ndarray, size: int = 160) -> np.ndarray:
    """
    Convert BGR uint8 -> RGB float32 and resize to FaceNet's expected input size.
    Here we scale to [0,1]; adjust if your model expects [-1,1] or mean/std.
    """
    img = cv2.resize(img_bgr, (size, size))
    img = img[..., ::-1].astype(np.float32)  # BGR -> RGB
    img = img / 255.0
    return img

def embed_face(sess: tf.Session, tensors: dict, face_rgb_float: np.ndarray) -> np.ndarray:
    """
    Run one aligned face through FaceNet and return a single embedding vector.
    Input must be RGB float32, shape (H, W, C).
    """
    batch = np.expand_dims(face_rgb_float, axis=0)           # [1, H, W, C]
    feed = {tensors["input"]: batch, tensors["phase"]: False}
    emb = sess.run(tensors["emb"], feed_dict=feed)           # [1, D]
    return emb[0]                                            # [D]

# ---------- Align + predict ----------
def detect_and_align(
    image_path: str,
    aligner: AlignDlib,
    image_size: int = 160,
    upsample_times: int = 1
) -> Optional[np.ndarray]:
    """
    Read an image, pick the largest face, align it to a square crop.
    Returns aligned BGR image (uint8) or None if no face found.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"unreadable: {image_path}")
        return None

    bb = aligner.get_largest_bb(img, upsample_times=upsample_times)
    if bb is None:
        print(f"no face: {image_path}")
        return None

    aligned = aligner.align(img, bb, output_size=image_size)
    return aligned

def top_k_predictions(
    probs: np.ndarray, class_names: List[str], k: int = 3
) -> List[Tuple[str, float]]:
    """
    Convert probability vector -> list of (class_name, prob) sorted desc, top-k.
    """
    idxs = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idxs]

# ---------- CLI pipeline ----------
def main(args):
    # 1) Init aligner (landmark predictor)
    aligner = AlignDlib(args.landmark_path)

    # 2) Detect + align the face from the input image
    aligned_bgr = detect_and_align(
        args.image_path, aligner, image_size=args.image_size, upsample_times=args.upsample
    )
    if aligned_bgr is None:
        raise SystemExit("Could not align a face from the input image.")

    # 3) Load FaceNet model
    g, sess, t = load_facenet(args.model_path)

    # 4) Convert aligned face to FaceNet input, then get the embedding
    aligned_rgb = bgr_to_facenet_rgb(aligned_bgr, size=args.image_size)
    with sess.as_default(), g.as_default():
        emb = embed_face(sess, t, aligned_rgb)   # shape [D]

    # 5) Load the trained classifier
    with open(args.classifier_pickle, "rb") as f:
        ckpt = pickle.load(f)
    clf = ckpt["classifier"]
    class_names = ckpt["class_names"]

    # 6) Predict probabilities and print top-k
    probs = clf.predict_proba([emb])[0]  # shape [num_classes]
    topk = top_k_predictions(probs, class_names, k=args.top_k)

    print("\nPrediction (top-{}):".format(args.top_k))
    for name, p in topk:
        print(f"  {name:20s}  {p: .4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--landmark_path", required=True)       # shape_predictor_68_face_landmarks.dat
    ap.add_argument("--model_path", required=True)          # FaceNet .pb
    ap.add_argument("--classifier_pickle", required=True)   # pickled SVM from training
    ap.add_argument("--image_size", type=int, default=160)
    ap.add_argument("--upsample", type=int, default=1)
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()
    main(args)
