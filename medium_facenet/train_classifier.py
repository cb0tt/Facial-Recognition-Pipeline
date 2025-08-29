"""
Run FaceNet on aligned images to get embeddings, then train a classifier (e.g., SVM).
Saves classifier to disk
"""

# medium_facenet_tutorial/train_classifier.py
import os, argparse, pickle, glob
import numpy as np, cv2, tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter

# ---------- helpers ----------
def list_images(root):
    exts = {".jpg", ".jpeg", ".png"}
    paths, classes = [], []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        for p in glob.glob(os.path.join(d, "*")):
            if os.path.splitext(p.lower())[1] in exts:
                paths.append(p); classes.append(cls)
    return paths, classes

def make_labels(classes):
    names = sorted(set(classes))
    name_to_idx = {n:i for i,n in enumerate(names)}
    y = np.array([name_to_idx[c] for c in classes], dtype=np.int32)
    return y, names

def load_facenet(pb_path):
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session(graph=g)
        with tf.gfile.GFile(pb_path, "rb") as f:
            gd = tf.GraphDef(); gd.ParseFromString(f.read())
            tf.import_graph_def(gd, name="")
        input_t = g.get_tensor_by_name("input:0")
        emb_t   = g.get_tensor_by_name("embeddings:0")
        phase_t = g.get_tensor_by_name("phase_train:0")
    return g, sess, {"input": input_t, "emb": emb_t, "phase": phase_t}

def read_img(path, size=160):
    img = cv2.imread(path)
    if img is None: raise ValueError(path)
    img = cv2.resize(img, (size, size))
    img = img[..., ::-1].astype(np.float32) / 255.0
    return img

def embed_paths(sess, tensors, paths, batch=64, size=160):
    X = []
    for i in range(0, len(paths), batch):
        b = [read_img(p, size=size) for p in paths[i:i+batch]]
        b = np.stack(b, axis=0)
        feed = {tensors["input"]: b, tensors["phase"]: False}
        E = sess.run(tensors["emb"], feed_dict=feed)
        X.append(E)
    return np.vstack(X) if X else np.zeros((0, 128), np.float32)

# ---------- main ----------
def main(args):
    paths, cls = list_images(args.aligned_dir)
    if not paths:
        raise SystemExit(f"No images found in: {args.aligned_dir}")

    # Drop classes with < min_per_class images
    counts = Counter(cls)
    keep = {c for c, n in counts.items() if n >= args.min_per_class}
    filtered = [(p, c) for p, c in zip(paths, cls) if c in keep]
    if not filtered:
        raise SystemExit("No classes left after filtering. Try lowering --min_per_class.")
    paths, cls = zip(*filtered)
    paths, cls = list(paths), list(cls)

    y, class_names = make_labels(cls)

    g, sess, t = load_facenet(args.model_path)
    with sess.as_default(), g.as_default():
        X = embed_paths(sess, t, paths, batch=args.batch, size=args.image_size)

    if args.is_train:
        # Train SVM
        clf = SVC(kernel="linear", probability=True)
        clf.fit(X, y)
        os.makedirs(os.path.dirname(args.out_pickle) or ".", exist_ok=True)
        with open(args.out_pickle, "wb") as f:
            pickle.dump({"classifier": clf, "class_names": class_names}, f)
        print("Trained on", len(paths), "images across", len(class_names), "classes")
        print("Saved:", args.out_pickle)
        print("classes:", class_names)  # <- prints full class list
    else:
        # Evaluate with existing classifier (compare NAMES, not indices)
        with open(args.out_pickle, "rb") as f:
            ckpt = pickle.load(f)
        clf = ckpt["classifier"]
        train_class_names = ckpt["class_names"]
        train_name_to_idx = {n: i for i, n in enumerate(train_class_names)}

        # Keep only eval samples whose class exists in the TRAINED classifier
        keep_mask = [c in train_name_to_idx for c in cls]
        paths_kept = [p for p, m in zip(paths, keep_mask) if m]
        cls_kept   = [c for c, m in zip(cls, keep_mask) if m]
        if not paths_kept:
            raise SystemExit("No eval images match any trained class names. "
                             "Ensure folder names in aligned_dir match training classes exactly.")

        # Re-embed kept eval images
        with sess.as_default(), g.as_default():
            X_eval = embed_paths(sess, t, paths_kept, batch=args.batch, size=args.image_size)

        # Predict -> names
        pred_idx  = clf.predict(X_eval)  # indices in training space
        pred_name = [train_class_names[i] for i in pred_idx]

        # Compare true name vs predicted name
        y_true_name = np.array(cls_kept)
        y_pred_name = np.array(pred_name)
        acc = (y_true_name == y_pred_name).mean()

        print(f"Evaluated on {len(y_true_name)} images "
              f"(kept {len(paths_kept)} that matched trained classes)")
        print("Overall accuracy:", f"{acc:.4f}")

        # Simple per-class accuracy (only over kept classes)
        from collections import defaultdict
        correct, total = defaultdict(int), defaultdict(int)
        for tname, pname in zip(y_true_name, y_pred_name):
            total[tname] += 1
            if tname == pname:
                correct[tname] += 1

        print("\nSample per-class accuracy (first 10 classes):")
        shown = 0
        for cname in sorted(set(y_true_name)):
            if shown >= 10: break
            if total[cname]:
                print(f"{cname:30s} {correct[cname]/total[cname]:.3f} ({total[cname]} samples)")
                shown += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_pickle", default="etc/facenet_svm.pkl")
    ap.add_argument("--image_size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min_per_class", type=int, default=10,
                    help="Drop identities with fewer than this many images")
    ap.add_argument("--is_train", action="store_true", help="train mode (default eval)")
    args = ap.parse_args()
    main(args)
