"""
align_dlib.py
--------------
Wraps dlib face detection and landmark alignment.
- load the dlib face detector
- load the shape predictor (shape_predictor_68_face_landmarks.dat)
- detect faces in an image
- return aligned face(s)
"""


import dlib
import cv2
import numpy as np

class AlignDlib:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_path)

    def _to_gray(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return gray.astype(np.uint8)

    @staticmethod
    def _area(rect):
        return max(0, rect.width()) * max(0, rect.height())

    @staticmethod
    def rect_to_xyxy(rect):
        return rect.left(), rect.top(), rect.right(), rect.bottom()

    def detect(self, img, upsample_times=1, score_threshold=0.0, return_scores=False):
        gray = self._to_gray(img)
        dets, scores, _ = self.detector.run(gray, upsample_times, score_threshold)
        pairs = [(r, s) for r, s in zip(dets, scores) if s >= score_threshold]
        pairs.sort(key=lambda rs: rs[1], reverse=True)
        rects = [r for r, _ in pairs]
        return (rects, [s for _, s in pairs]) if return_scores else rects

    def get_largest_bb(self, img, upsample_times=1):
        rects = self.detect(img, upsample_times=upsample_times)
        if not rects:
            return None
        return max(rects, key=lambda r: r.width() * r.height())

    def landmarks(self, img, bb):
        gray = self._to_gray(img)
        return self.shape_predictor(gray, bb)

    def align(self, img, bb, output_size=160, padding=0.25):
        shape = self.landmarks(img, bb)
        if shape is None:
            return None

        left_eye  = (shape.part(36).x, shape.part(36).y)
        right_eye = (shape.part(45).x, shape.part(45).y)
        nose_tip  = (shape.part(33).x, shape.part(33).y)

        ref_pts = np.float32([
            [0.3 * output_size, 0.35 * output_size],  # left eye
            [0.7 * output_size, 0.35 * output_size],  # right eye
            [0.5 * output_size, 0.55 * output_size],  # nose tip
        ])
        dst_pts = np.float32([left_eye, right_eye, nose_tip])

        M = cv2.getAffineTransform(dst_pts, ref_pts)
        aligned = cv2.warpAffine(img, M, (output_size, output_size))
        return aligned










