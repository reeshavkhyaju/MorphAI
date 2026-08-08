import cv2
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_IMG_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MANIFEST_DIR, IMG_SIZE

def resize_split(split_name, dst_dir):
    filelist_path = os.path.join(MANIFEST_DIR, f"filelist_{split_name}.txt")
    with open(filelist_path) as f:
        filenames = [l.strip() for l in f if l.strip()]

    skipped = []
    for fname in tqdm(filenames, desc=f"Resizing {split_name}"):
        src_path = os.path.join(RAW_IMG_DIR, fname)
        img = cv2.imread(src_path)
        if img is None:
            skipped.append(fname)
            continue
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dst_dir, fname), img_resized)

    print(f"{split_name}: skipped {len(skipped)} unreadable images")
    return skipped

if __name__ == "__main__":
    resize_split("train", TRAIN_DIR)
    resize_split("val", VAL_DIR)
    resize_split("test", TEST_DIR)