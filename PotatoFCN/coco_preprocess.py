import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

# --- 설정 ---
# COCO 2017 데이터셋 경로
DATA_DIR = './coco_data/coco2017/'
DATA_TYPE = 'train2017'

# 전처리된 파일을 저장할 경로
OUTPUT_DIR = './preprocessed_data'
# 모델에 입력할 이미지 크기
IMG_SIZE = (128, 128)
# 처리할 이미지 개수 (전체는 약 11만장으로 매우 오래 걸리므로, 테스트용으로 줄여서 사용)
NUM_SAMPLES_TO_PROCESS = 5000

# --- 전처리 시작 ---
ann_file = os.path.join(DATA_DIR, 'annotations', f'instances_{DATA_TYPE}.json')
img_dir = os.path.join(DATA_DIR, DATA_TYPE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

coco = COCO(ann_file)
img_ids = coco.getImgIds()[:NUM_SAMPLES_TO_PROCESS]

all_images = []
all_masks = []

print(f"Processing {len(img_ids)} images...")

for img_id in tqdm(img_ids):
    # 이미지 정보 로드
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    
    # 이미지 읽기 및 흑백 변환
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        continue
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 해당 이미지의 모든 분할 마스크를 하나로 합치기
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    combined_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
    for ann in anns:
        # coco.annToMask는 개별 객체 마스크를 생성
        combined_mask = np.maximum(combined_mask, coco.annToMask(ann))

    # 이미지와 마스크를 지정된 크기로 리사이즈
    resized_image = cv2.resize(gray_image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(combined_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

    # 0~1 사이 값으로 정규화
    normalized_image = resized_image.astype(np.float32) / 255.0
    
    all_images.append(normalized_image)
    all_masks.append(resized_mask)

# 리스트를 Numpy 배열로 변환하고 1차원으로 펼치기
images_array = np.array(all_images, dtype=np.float32).flatten()
masks_array = np.array(all_masks, dtype=np.float32).flatten()

# 바이너리 파일로 저장
images_array.tofile(os.path.join(OUTPUT_DIR, 'coco_images.bin'))
masks_array.tofile(os.path.join(OUTPUT_DIR, 'coco_masks.bin'))

# 데이터 정보 저장 (C에서 읽기 위함)
with open(os.path.join(OUTPUT_DIR, 'info.txt'), 'w') as f:
    f.write(f"{len(all_images)}\n")
    f.write(f"{IMG_SIZE[0]}\n")
    f.write(f"{IMG_SIZE[1]}\n")

print("\nPreprocessing finished!")
print(f"Saved {len(all_images)} samples to '{OUTPUT_DIR}'")