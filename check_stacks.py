from pathlib import Path
from collections import defaultdict
import cv2
from tqdm import tqdm
import utils
import csv

# base path
# BASE = Path("O:/Data-Work/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/")
BASE = Path("/agroscope/Data-Work-CH/22_Plant_Production-CH/224_Digitalisation/Jonas_Anderegg_Files/B_Data/03_PreDiMix/Uitikon/")


# # =================================================================================================
# # Ensure that a multiple of 15 images is present for each plot
# # =================================================================================================

# # get paths
# folders = []

# for sub_dir in BASE.iterdir():
#     if not sub_dir.is_dir():
#         continue

#     for camera_dir in sub_dir.iterdir():
#         if camera_dir.exists():
#             # collect only directories whose name starts with 8 digits (YYYYMMDD)
#             # and exclude any that contain "Leaf" (case-insensitive)
#             for p in camera_dir.iterdir():
#                 if not p.is_dir():
#                     continue
#                 name = p.name
#                 if not re.match(r'^\d{8}', name):
#                     continue
#                 if 'leaf' in name.lower():
#                     continue
#                 folders.append(p)


# problem_folders = []

# for folder in tqdm(folders, desc="Processing folders"):

#     # Only keep folders whose name starts with CHWW
#     if not re.match(r'^\d{8}', folder.name):
#         continue

#     # Count JPG files (adjust pattern if needed)
#     n_files = len(list(folder.glob("*.JPG")))

#     if n_files % 15 != 0:
#         problem_folders.append({
#             "folder": str(folder),
#             "n_files": n_files,
#             "remainder": n_files % 15
#         })

# # Save log
# log_file = BASE / "folders_with_incorrect_file_count.csv"

# pd.DataFrame(problem_folders).to_csv(
#     log_file,
#     index=False
# )

# =================================================================================================
# Try to extract the marker from the first image of a stack and save it to a new folder
# =================================================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

def get_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    )


csv_path = BASE / "marker_scores.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["image_path", "score"])


for camera_dir in BASE.rglob("Camera*"):

    if not camera_dir.parents[0].name == "20260622_Uitikon":
        continue

    print(f'PROCESSING: ', {camera_dir})

    # Group directories by their last 5 characters
    groups = defaultdict(list)

    for plot_dir in tqdm(camera_dir.iterdir(), desc="Processing plots"):

        if not plot_dir.is_dir():
            continue

        # Ignore things such as $RECYCLE.BIN
        if len(plot_dir.name) < 5:
            continue

        group_id = plot_dir.name[-5:]
        groups[group_id].append(plot_dir)

    # Process each plot group
    for group_id, plot_dirs in groups.items():

        # if not group_id == '00778':
        #     continue

        # Combine images from all folders belonging to this group
        images = []

        for plot_dir in sorted(plot_dirs):
            images.extend(get_images(plot_dir))

        # Sample images 1, 16, 31, 46, ...
        sampled_images = images[0::15]

        # print(f"\nGroup: {group_id}")
        # print(f"Folders: {[p.name for p in plot_dirs]}")
        # print(f"Total images: {len(images)}")
        # print(f"Sampled: {len(sampled_images)}")

        for image_path in sampled_images:
            # print(image_path.name)
            # if not image_path.name == '1H2A0919.JPG':
            #     continue    
            img = cv2.imread(str(image_path))
            marker, score = utils.find_bw_marker(img)

            # append score and image path to CSV continuously
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([str(image_path), float(score)])

            m = cv2.resize(marker, (100, 100), interpolation=cv2.INTER_AREA)
            output_path = image_path.parents[2] / "markers" / f"marker_{image_path.parents[0].name}_{image_path.name}"
            output_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(output_path), cv2.cvtColor(m, cv2.COLOR_BGR2RGB))

# =====> COPY ALL OUTPUT IMAGES WITH INCORRECT DETECTIONS FROM THE "markers" FOLDER TO A NEW FOLDER PER SITE_DATE


