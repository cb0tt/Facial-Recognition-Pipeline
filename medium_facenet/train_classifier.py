"""
Run FaceNet on aligned images to get embeddings, then train a classifier (e.g., SVM).

Train mode (--is_train):
- Reads images from --aligned_dir/<person>/*, embeds them with FaceNet, trains an SVM, saves to --out_pickle.

Eval mode (default):
- Embeds ALL images under --aligned_dir (recursive) and prints SVM predictions + probabilities.
"""


import os, argparse, pickle, glob, csv
import numpy as np, cv2, tensorflow as tf
from sklearn.svm import SVC
from collections import Counter


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

def list_images_recursive(root):
    exts = {".jpg", ".jpeg", ".png"}
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                paths.append(os.path.join(dirpath, f))
    paths.sort()
    return paths

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

def top_k_predictions(probs, class_names, k=3):
    k = int(k)
    if k <= 0:
        return []
    k = min(k, len(class_names))
    idxs = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idxs]

def main(args):
    if args.is_train:
        paths, cls = list_images(args.aligned_dir)
        if not paths:
            raise SystemExit(f"No images found in: {args.aligned_dir}")

        # Drop classes with < min_per_class images (training-time only)
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

        # Train SVM
        clf = SVC(kernel="linear", probability=True)
        clf.fit(X, y)
        os.makedirs(os.path.dirname(args.out_pickle) or ".", exist_ok=True)
        with open(args.out_pickle, "wb") as f:
            pickle.dump({"classifier": clf, "class_names": class_names}, f)
        print("Trained on", len(paths), "images across", len(class_names), "classes")
        print("Saved:", args.out_pickle)
        print("classes:", class_names) 
    else:
        with open(args.out_pickle, "rb") as f:
            ckpt = pickle.load(f)
        clf = ckpt["classifier"]
        train_class_names = ckpt["class_names"]

        g, sess, t = load_facenet(args.model_path)
        with sess.as_default(), g.as_default():
            paths = list_images_recursive(args.aligned_dir)
            if not paths:
                raise SystemExit(f"No images found in: {args.aligned_dir}")

            X_eval = embed_paths(sess, t, paths, batch=args.batch, size=args.image_size)
            probs = clf.predict_proba(X_eval)

        rows = []
        for p, pr in zip(paths, probs):
            topk = top_k_predictions(pr, train_class_names, k=args.top_k)
            best_name, best_p = topk[0] if topk else ("", float("nan"))
            rows.append((p, best_name, best_p, topk))

        print(f"Predicted {len(rows)} images from: {args.aligned_dir}")
        for p, best_name, best_p, topk in rows:
            rel = os.path.relpath(p, args.aligned_dir)
            topk_str = ", ".join([f"{n}:{pp:.4f}" for n, pp in topk])
            print(f"{rel} -> {best_name} ({best_p:.4f})  [{topk_str}]")

        if args.out_csv:
            os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["path", "predicted_name", "predicted_prob", "top_k"])
                for p, best_name, best_p, topk in rows:
                    topk_str = ";".join([f"{n}:{pp:.6f}" for n, pp in topk])
                    w.writerow([p, best_name, f"{best_p:.6f}", topk_str])
            print("Wrote:", args.out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_pickle", default="etc/facenet_svm.pkl")
    ap.add_argument("--image_size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min_per_class", type=int, default=10,
                    help="Drop identities with fewer than this many images")
    ap.add_argument("--predict_only", action="store_true",
                    help="(Deprecated; no-op) Eval mode always outputs predictions for images in aligned_dir (recursive).")
    ap.add_argument("--top_k", type=int, default=3,
                    help="Eval mode: show top-k classes per image.")
    ap.add_argument("--out_csv", default="",
                    help="Eval mode: optional path to write predictions as CSV.")
    ap.add_argument("--is_train", action="store_true", help="train mode (default eval)")
    args = ap.parse_args()
    main(args)
