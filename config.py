import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_IMG_DIR = os.path.join(RAW_DIR, "img_align_celeba", "img_align_celeba")
PARTITION_CSV = os.path.join(RAW_DIR, "list_eval_partition.csv")

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DIR, "test")
MASK_DIR = os.path.join(PROCESSED_DIR, "masks")
HEATMAP_DIR = os.path.join(PROCESSED_DIR, "heatmaps")
MANIFEST_DIR = os.path.join(PROCESSED_DIR, "manifests")

IMG_SIZE = 256

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, MASK_DIR, HEATMAP_DIR, MANIFEST_DIR]:
    os.makedirs(d, exist_ok=True)

LGNET_REPO_DIR = os.path.join(BASE_DIR, "external", "lgnet")
LGNET_WEIGHTS_PATH = os.path.join(LGNET_REPO_DIR, "checkpoints", "celebahq_LGNet", "latest_net_G1.pth")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 8
NUM_WORKERS = 4