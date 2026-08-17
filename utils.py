import cv2
import numpy as np
from matplotlib import pyplot as plt


def neutral_mask(image, saturation_threshold=30):
    """
    Identify approximately black/white/gray pixels.
    image: RGB uint8 image
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Low saturation = little/no color
    mask = hsv[:, :, 1] < saturation_threshold

    return mask


def bimodal_score(roi):

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_RGB2HSV
    )

    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    threshold, _ = cv2.threshold(
        val.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    dark = val < threshold
    bright = ~dark

    if not dark.any() or not bright.any():
        return 0

    # Similar class sizes
    balance = 2 * min(
        dark.mean(),
        bright.mean()
    )

    # Color variability within each class
    color_variability = (
        sat[dark].std()
        + sat[bright].std()
    ) / 2

    # Brightness separation
    brightness_separation = (
        abs(
            val[dark].mean()
            - val[bright].mean()
        )
    )

    # High separation + low within-class variability
    score = (
        balance
        * brightness_separation
        / (color_variability + 1)
    )

    return score


def find_bw_marker(
    image,
    corner_fraction=0.1,
    min_area_fraction=0.005,
    max_area_fraction=0.5,
):

    """
    Search for a black-and-white marker in one of the four image corners.

    Returns
    -------
    marker : dict or None
        {
            "corner": "top_left",
            "bbox": (x, y, w, h),
            "contour": contour,
            "roi": roi
        }
    """

    h, w = image.shape[:2]

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    roi_h = int(h * corner_fraction)
    roi_w = int(w * corner_fraction)

    inset_w = int(w * 0.05)
    inset_h = int(h * 0.05)

    corners = {
        "top_left": (
            inset_w,
            inset_h,
            inset_w + roi_w,
            inset_h + roi_h,
        ),

        "top_right": (
            w - inset_w - roi_w,
            inset_h,
            w - inset_w,
            inset_h + roi_h,
        ),

        "bottom_left": (
            inset_w,
            h - inset_h - roi_h,
            inset_w + roi_w,
            h - inset_h,
        ),

        "bottom_right": (
            w - inset_w - roi_w,
            h - inset_h - roi_h,
            w - inset_w,
            h - inset_h,
        ),
    }

    image_area = h * w

    rois = []
    local_variances = []
    scores = []

    for corner, (x1, y1, x2, y2) in corners.items():

        roi = rgb_image[y1:y2, x1:x2]
        rois.append(roi)

        # ---------------------------------------------------------
        # Convert to HSV
        # ---------------------------------------------------------

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_RGB2HSV
        )

        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        # ---------------------------------------------------------
        # Local brightness contrast
        # ---------------------------------------------------------

        val_mean = cv2.GaussianBlur(
            val,
            (0, 0),
            5
        )

        val_contrast = np.abs(
            val - val_mean
        )

        # ---------------------------------------------------------
        # Local saturation contrast
        # ---------------------------------------------------------

        sat_mean = cv2.GaussianBlur(
            sat,
            (0, 0),
            5
        )

        sat_contrast = np.abs(
            sat - sat_mean
        )

        # ---------------------------------------------------------
        # Marker score map
        # ---------------------------------------------------------

        score_map = (
            val_contrast /
            (sat_contrast + 10)
        )

        score_map_smooth = cv2.GaussianBlur(
            score_map,
            (0, 0),
            2
        )

        # ---------------------------------------------------------
        # Regional variability
        # ---------------------------------------------------------

        window_sigma = 8

        local_mean = cv2.GaussianBlur(
            score_map_smooth,
            (0, 0),
            window_sigma
        )

        local_mean_sq = cv2.GaussianBlur(
            score_map_smooth ** 2,
            (0, 0),
            window_sigma
        )

        local_variance = (
            local_mean_sq -
            local_mean ** 2
        )

        local_variance = np.maximum(
            local_variance,
            0
        )

        # ---------------------------------------------------------
        # Green mask
        # ---------------------------------------------------------

        hue = hsv[:, :, 0]

        green = (
            (hue > 30) &
            (hue < 100) &
            (sat > 40)
        )

        # Calculate green fraction at the same spatial scale
        # as the regional variability
        green_fraction = cv2.GaussianBlur(
            green.astype(np.float32),
            (0, 0),
            window_sigma
        )

        # ---------------------------------------------------------
        # Penalize green regions
        # ---------------------------------------------------------

        regional_score = (
            local_variance *
            (1 - green_fraction)
        )

        # Store the original local variance for visualization
        local_variances.append(
            local_variance
        )

        # ---------------------------------------------------------
        # Score for selecting the best corner
        # ---------------------------------------------------------

        scores.append(
            np.percentile(
                regional_score,
                95
            )
        )

    score_threshold = 0.03

    candidate_indices = [
        i for i, score in enumerate(scores)
        if score >= score_threshold
    ]

    if len(candidate_indices) >= 2:

        bimodal_scores = []

        for i in candidate_indices:

            roi = rois[i]

            hsv = cv2.cvtColor(
                roi,
                cv2.COLOR_RGB2HSV
            )

            sat = hsv[:, :, 1].astype(np.float32)
            val = hsv[:, :, 2].astype(np.float32)

            # -------------------------------------------------
            # Split brightness into two groups using Otsu
            # -------------------------------------------------

            threshold, _ = cv2.threshold(
                val.astype(np.uint8),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            dark = val < threshold
            bright = ~dark

            if not dark.any() or not bright.any():
                bimodal_scores.append(0)
                continue

            # -------------------------------------------------
            # 1. Similar class sizes
            # -------------------------------------------------

            balance = 2 * min(
                dark.mean(),
                bright.mean()
            )

            # -------------------------------------------------
            # 2. Color variability within each class
            # -------------------------------------------------

            color_variability = (
                sat[dark].std()
                + sat[bright].std()
            ) / 2

            # -------------------------------------------------
            # Final bimodality score
            # -------------------------------------------------

            bimodality = (
                balance /
                (color_variability + 1)
            )

            bimodal_scores.append(bimodality)

        # Pick candidate with best bimodality
        best_idx = candidate_indices[
            np.argmax(bimodal_scores)
        ]

        best_corner = list(corners.keys())[best_idx]


    else:

        best_idx = np.argmax(scores)

        best_corner = list(
            corners.keys()
        )[best_idx]

    print("Selected:", best_corner)
    print("Score:", scores[best_idx])

    # =============================================================
    # Common colour scale
    # =============================================================

    all_values = np.concatenate([
        lv.ravel()
        for lv in local_variances
    ])

    vmin = np.percentile(
        all_values,
        1
    )

    vmax = np.percentile(
        all_values,
        99
    )    

    # # =============================================================
    # # Plot all four
    # # =============================================================

    # fig, axes = plt.subplots(
    #     2,
    #     4,
    #     figsize=(16, 8)
    # )

    # for i, corner in enumerate(corners.keys()):

    #     selected = i == best_idx

    #     # ---------------------------------------------------------
    #     # RGB
    #     # ---------------------------------------------------------

    #     axes[0, i].imshow(rois[i])

    #     axes[0, i].set_title(
    #         f"{corner}\nscore = {scores[i]:.3f}",
    #         fontweight="bold" if selected else "normal"
    #     )

    #     axes[0, i].axis("off")

    #     # Red border around selected ROI
    #     if selected:
    #         for spine in axes[0, i].spines.values():
    #             spine.set_visible(True)
    #             spine.set_linewidth(4)
    #             spine.set_edgecolor("red")

    #     # ---------------------------------------------------------
    #     # Local variance
    #     # ---------------------------------------------------------

    #     im = axes[1, i].imshow(
    #         local_variances[i],
    #         cmap="magma",
    #         vmin=vmin,
    #         vmax=vmax
    #     )

    #     axes[1, i].set_title(
    #         "Regional variability",
    #         fontweight="bold" if selected else "normal"
    #     )

    #     axes[1, i].axis("off")

    #     # Red border around selected variance map
    #     if selected:
    #         for spine in axes[1, i].spines.values():
    #             spine.set_visible(True)
    #             spine.set_linewidth(4)
    #             spine.set_edgecolor("red")


    # # Shared colourbar
    # fig.colorbar(
    #     im,
    #     ax=axes[1, :],
    #     fraction=0.02,
    #     pad=0.02,
    #     label="Local variance"
    # )

    # plt.tight_layout()
    # plt.show(block=True)

    return rois[best_idx], scores[best_idx]
    