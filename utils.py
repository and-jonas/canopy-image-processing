import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_marker(image, corner_fraction=0.1, min_area_fraction=0.005, max_area_fraction=0.5):
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    roi_h, roi_w = int(h * corner_fraction), int(w * corner_fraction)
    inset_w, inset_h = int(w * 0.05), int(h * 0.05)
    corners = {
        "top_left": (inset_w, inset_h, inset_w + roi_w, inset_h + roi_h),
        "top_right": (w - inset_w - roi_w, inset_h, w - inset_w, inset_h + roi_h),
        "bottom_left": (inset_w, h - inset_h - roi_h, inset_w + roi_w, h - inset_h),
        "bottom_right": (w - inset_w - roi_w, h - inset_h - roi_h, w - inset_w, h - inset_h),
    }

    rois, local_variances, scores = [], [], []
    for _, (x1, y1, x2, y2) in corners.items():
        roi = rgb[y1:y2, x1:x2]
        rois.append(roi)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        val_mean = cv2.GaussianBlur(val, (0, 0), 5)
        val_contrast = np.abs(val - val_mean)
        sat_mean = cv2.GaussianBlur(sat, (0, 0), 5)
        sat_contrast = np.abs(sat - sat_mean)
        score_map = val_contrast / (sat_contrast + 10)
        score_map_smooth = cv2.GaussianBlur(score_map, (0, 0), 2)
        window_sigma = 8
        local_mean = cv2.GaussianBlur(score_map_smooth, (0, 0), window_sigma)
        local_mean_sq = cv2.GaussianBlur(score_map_smooth ** 2, (0, 0), window_sigma)
        local_variance = np.maximum(local_mean_sq - local_mean ** 2, 0)
        hue = hsv[:, :, 0]
        green = (hue > 30) & (hue < 100) & (sat > 40)
        green_fraction = cv2.GaussianBlur(green.astype(np.float32), (0, 0), window_sigma)
        regional_score = local_variance * (1 - green_fraction)
        local_variances.append(local_variance)
        scores.append(np.percentile(regional_score, 95))

    score_threshold = 0.03
    candidate_indices = [i for i, s in enumerate(scores) if s >= score_threshold]
    if len(candidate_indices) >= 2:
        bimodal_scores = []
        for i in candidate_indices:
            roi = rois[i]
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1].astype(np.float32)
            val = hsv[:, :, 2].astype(np.float32)
            thresh, _ = cv2.threshold(val.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark = val < thresh
            bright = ~dark
            if not dark.any() or not bright.any():
                bimodal_scores.append(0)
                continue
            balance = 2 * min(dark.mean(), bright.mean())
            color_variability = (sat[dark].std() + sat[bright].std()) / 2
            bimodal_scores.append(balance / (color_variability + 1))
        best_idx = candidate_indices[int(np.argmax(bimodal_scores))]
    else:
        best_idx = int(np.argmax(scores))

    return rois[best_idx], scores[best_idx]
    