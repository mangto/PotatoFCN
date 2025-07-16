import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import random

# --- 설정 ---
DATA_DIR = './coco_data/coco2017/'
DATA_TYPE = 'train2017'
OUTPUT_DIR = './preprocessed_data'
IMG_SIZE = (128, 128)
NUM_SAMPLES_TO_PROCESS = 5
AUGMENTATIONS_PER_IMAGE = 0

ann_file = os.path.join(DATA_DIR, 'annotations', f'instances_{DATA_TYPE}.json')
img_dir = os.path.join(DATA_DIR, DATA_TYPE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

coco = COCO(ann_file)
img_ids = coco.getImgIds()[:NUM_SAMPLES_TO_PROCESS]

all_images = []
all_masks = []

def apply_augmentation(img, mask):
    # 1. Random horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    # 2. Random brightness shift
    if random.random() < 0.5:
        shift = random.uniform(-0.2, 0.2)
        img = np.clip(img + shift * 255, 0, 255)

    # 3. Random scale & center crop
    if random.random() < 0.5:
        scale = random.uniform(1.0, 1.2)
        h, w = img.shape[:2]
        resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        resized_mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

        # crop to center
        start_x = (resized_img.shape[1] - w) // 2
        start_y = (resized_img.shape[0] - h) // 2
        img = resized_img[start_y:start_y+h, start_x:start_x+w]
        mask = resized_mask[start_y:start_y+h, start_x:start_x+w]

    return img, mask

print(f"Processing {len(img_ids)} images...")

for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        continue
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    combined_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
    for ann in anns:
        combined_mask = np.maximum(combined_mask, coco.annToMask(ann))

    # Resize to fixed size first (before augmentation)
    gray_image = cv2.resize(gray_image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    combined_mask = cv2.resize(combined_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

    # Normalize original image
    normalized_image = gray_image.astype(np.float32) / 255.0
    all_images.append(normalized_image)
    all_masks.append(combined_mask)

    # --- Augmentation ---
    for _ in range(AUGMENTATIONS_PER_IMAGE):
        aug_img, aug_mask = apply_augmentation(gray_image.copy(), combined_mask.copy())
        aug_img = cv2.resize(aug_img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        aug_mask = cv2.resize(aug_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        normalized_aug = aug_img.astype(np.float32) / 255.0

        all_images.append(normalized_aug)
        all_masks.append(aug_mask)

images_array = np.array(all_images, dtype=np.float32).flatten()
masks_array = np.array(all_masks, dtype=np.float32).flatten()

images_array.tofile(os.path.join(OUTPUT_DIR, 'coco_images.bin'))
masks_array.tofile(os.path.join(OUTPUT_DIR, 'coco_masks.bin'))

with open(os.path.join(OUTPUT_DIR, 'info.txt'), 'w') as f:
    f.write(f"{len(all_images)}\n")
    f.write(f"{IMG_SIZE[0]}\n")
    f.write(f"{IMG_SIZE[1]}\n")

print("\nPreprocessing with augmentation finished!")
print(f"Saved {len(all_images)} samples to '{OUTPUT_DIR}'")
