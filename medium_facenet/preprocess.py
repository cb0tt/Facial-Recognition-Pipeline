"""
Read images from data/raw/<person>/*.jpg,
find LARGEST face, align it, and write to data/aligned/<person>/.
"""

import os
import argparse
import cv2
from align_dlib import AlignDlib

def iter_images(root):
    exts = {".jpg", ".jpeg", ".png"}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                yield os.path.join(dirpath, f)



def preprocess(input_dir, output_dir, landmark_path, image_size=160, upsample=1):
    os.makedirs(output_dir, exist_ok=True)
    aligner = AlignDlib(landmark_path)

    for person in sorted(os.listdir(input_dir)):
        in_p = os.path.join(input_dir, person)
        if not os.path.isdir(in_p):
            continue
        out_p = os.path.join(output_dir, person)
        os.makedirs(out_p, exist_ok=True)

        for img_path in iter_images(in_p):
            img = cv2.imread(img_path) #read image
            if img is None:
                print(f"skip unreadable: {img_path}")
                continue

            bb = aligner.get_largest_bb(img, upsample_times=upsample) #largest face
            if bb is None:
                print(f"no face: {img_path}")
                continue

            aligned = aligner.align(img, bb, output_size=image_size)
            if aligned is None:
                print(f"align fail: {img_path}")
                continue


            out_path = os.path.join(out_p, os.path.basename(img_path))
            cv2.imwrite(out_path, aligned)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--landmark_path", required=True)
    ap.add_argument("--image_size", type=int, default=160)
    ap.add_argument("--upsample", type=int, default=1)
    args = ap.parse_args()

    preprocess(
        args.input_dir,
        args.output_dir,
        args.landmark_path,
        image_size=args.image_size,
        upsample=args.upsample
    )