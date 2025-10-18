import gzip
import shutil
import os

data_dir = "C:/Users/ravin/OneDrive/Desktop/Coursework/DSAI-lab/Project/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

# Walk all subdirectories and decompress any .gz files in place
for root, _, files in os.walk(data_dir):
    for file_name in files:
        if not file_name.endswith(".gz"):
            continue

        src_path = os.path.join(root, file_name)

        # Preserve double extensions like .nii.gz
        if file_name.endswith(".nii.gz"):
            dest_name = file_name[:-3]  # remove only the trailing .gz
        else:
            dest_name, _ = os.path.splitext(file_name)

        dest_path = os.path.join(root, dest_name)

        # Skip if already decompressed
        if os.path.exists(dest_path):
            continue

        with gzip.open(src_path, "rb") as f_in, open(dest_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
