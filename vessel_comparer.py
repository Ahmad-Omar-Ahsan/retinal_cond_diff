import numpy as np
import cv2
import os
from glob import glob
import re

def dice_coefficient(mask1, mask2):
    """
    Compute the Dice Coefficient between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    return 2. * intersection / (mask1.sum() + mask2.sum() + 1e-8)

def load_binary_image(path):
    """
    Load an image in grayscale and binarize it.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return (img > 0).astype(np.uint8)

def extract_id(filename):
    match = re.search(r"__(\d+)_", filename)
    return match.group(1) if match else None

def match_files_by_id(dr_paths, normal_paths):
    normal_map = {extract_id(path): path for path in normal_paths}
    matched_pairs = []

    for dr_path in dr_paths:
        dr_id = extract_id(dr_path)
        if dr_id in normal_map:
            matched_pairs.append((dr_path, normal_map[dr_id]))
        else:
            print(f"No match found for DR image: {dr_path}")
    return matched_pairs

def compare_all(normal_dir, dr_dir, cohort_dir):
    normal_paths = sorted(glob(os.path.join(normal_dir, "*.png")))
    dr_paths = sorted(glob(os.path.join(dr_dir, "*.png")))
    cohort_paths = sorted(glob(os.path.join(cohort_dir, "*.png")))

    matched_pairs = match_files_by_id(dr_paths, normal_paths)

    top1_count = 0
    top3_count = 0
    top5_count = 0
    top10_count = 0
    flipped_dr_count = 0

    for idx, (normal_path, dr_path) in enumerate(matched_pairs, start=1):
        normal_mask = load_binary_image(normal_path)
        dr_mask = load_binary_image(dr_path)

        if normal_mask.shape != dr_mask.shape:
            print(f"Skipping {os.path.basename(normal_path)} due to shape mismatch.")
            continue

        # Try both original and flipped DR
        # flipped_dr = np.fliplr(dr_mask)
        best_dr_score = dice_coefficient(normal_mask, dr_mask)
        # dice_flipped = dice_coefficient(normal_mask, flipped_dr)

        # if dice_flipped > dice_orig:
        #     best_dr_score = dice_flipped
        #     dr_flipped_used = True
        #     flipped_dr_count += 1
        # else:
        #     best_dr_score = dice_orig
        #     dr_flipped_used = False

        dice_scores = [("dr_counterpart", best_dr_score)]

        for cohort_path in cohort_paths:
            cohort_mask = load_binary_image(cohort_path)
            if cohort_mask.shape != normal_mask.shape:
                continue
            dice = dice_coefficient(normal_mask, cohort_mask)
            dice_scores.append((os.path.basename(cohort_path), dice))

        ranked = sorted(dice_scores, key=lambda x: x[1], reverse=True)

        top_names = [name for name, _ in ranked]
        if "dr_counterpart" in top_names[:5]:
            top5_count += 1
        if "dr_counterpart" in top_names[:10]:
            top10_count += 1

        if "dr_counterpart" in top_names[:1]:
            top1_count += 1
        if "dr_counterpart" in top_names[:3]:
            top3_count += 1

        print(f"\n--- Normal Image {idx}: {os.path.basename(normal_path)} ---")
        # print(f"Top 5 closest vessel maps (DR flipped: {'Yes' if dr_flipped_used else 'No'}):")
        for rank, (name, score) in enumerate(ranked[:5], start=1):
            label = " (DR counterpart)" if name == "dr_counterpart" else ""
            print(f"{rank}. {name} - DICE: {score:.4f}{label}")

    print("\n==================== Summary ====================")
    print(f"DR counterfactual appeared in Top 1:  {top1_count} / {len(matched_pairs)}, {(top1_count * 100)/ len(matched_pairs):.2f}%")
    print(f"DR ccounterfactual appeared in Top 3: {top3_count} / {len(matched_pairs)}, {(top3_count * 100)/ len(matched_pairs):.2f}%")
    print(f"DR ccounterfactual appeared in Top 5:  {top5_count} / {len(matched_pairs)}, {(top5_count * 100)/ len(matched_pairs):.2f}%")
    print(f"DR counterfactual appeared in Top 10: {top10_count} / {len(matched_pairs)}, {(top10_count * 100)/ len(matched_pairs):.2f}%")
    # print(f"Flipped DR used:                  {flipped_dr_count} / {len(matched_pairs)}")

  

if __name__ == "__main__":
    dr_image_path = "/home/ahmad/ahmad_experiments/retinal_cond_diff/vessel_segmentations/DR/raw_binary"
    normal_image_path = "/home/ahmad/ahmad_experiments/retinal_cond_diff/vessel_segmentations/Normal_counterfactual/raw_binary"
    cohort_directory = "/home/ahmad/ahmad_experiments/retinal_cond_diff/vessel_segmentations/Normal_Cohort/raw_binary"

    results = compare_all(dr_image_path, normal_image_path, cohort_directory)
