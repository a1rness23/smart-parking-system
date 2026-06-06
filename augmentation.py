import cv2
import os
import numpy as np
import random

# ==========================================
# 1. 경로 설정 (VS Code 폴더 구조 기준)
# ==========================================
INPUT_DIR = 'rec_train_images'       # 💡 원본 이미지가 있는 폴더
OUTPUT_DIR = 'aug_train_images'      # 💡 증강된 이미지가 저장될 새 폴더
LABEL_FILE = 'train_data_draft.txt'  # 💡 올려주신 원본 라벨 파일명
NEW_LABEL_FILE = 'aug_train.txt'     # 💡 새로 만들어질 뻥튀기 라벨 파일명

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 증강 함수 (기하학적 & 환경적 변형)
# ==========================================
def adjust_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=(0, 0, 0)) 

def apply_blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0)

# ==========================================
# 3. 데이터 & 라벨 동시 뻥튀기 실행
# ==========================================
print(f"🚀 이미지 증강 및 새 라벨 생성을 시작합니다...")

# 원본 라벨 읽기
with open(LABEL_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 새 라벨 파일 쓰기 모드로 열기
with open(NEW_LABEL_FILE, 'w', encoding='utf-8') as out_f:
    count = 0
    for line in lines:
        line = line.strip()
        if not line: continue
            
        # 파일명과 정답 분리 (공백이나 탭 기준)
        parts = line.split()
        if len(parts) < 2: continue
            
        filename = parts[0]
        label = parts[1]
        
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ 경고: {filename} 이미지를 찾을 수 없어 건너뜁니다.")
            continue
            
        base_name, ext = os.path.splitext(filename)
        
        # 1. 원본
        name_orig = f"{base_name}_orig{ext}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, name_orig), img)
        out_f.write(f"{name_orig}\t{label}\n")
        
        # 2. 밝게
        name_bright = f"{base_name}_bright{ext}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, name_bright), adjust_brightness(img, 1.5))
        out_f.write(f"{name_bright}\t{label}\n")
        
        # 3. 어둡게
        name_dark = f"{base_name}_dark{ext}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, name_dark), adjust_brightness(img, 0.5))
        out_f.write(f"{name_dark}\t{label}\n")
        
        # 4. 미세 회전
        name_rot = f"{base_name}_rot{ext}"
        random_angle = random.uniform(-3.0, 3.0)
        cv2.imwrite(os.path.join(OUTPUT_DIR, name_rot), rotate_image(img, random_angle))
        out_f.write(f"{name_rot}\t{label}\n")
        
        # 5. 흐리게
        name_blur = f"{base_name}_blur{ext}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, name_blur), apply_blur(img))
        out_f.write(f"{name_blur}\t{label}\n")
        
        count += 1

print("-" * 50)
print(f"✅ 증강 완료! 원본 {count}장이 총 {count * 5}장으로 늘어났습니다.")
print(f"✅ 새로운 라벨 파일이 완성되었습니다: {NEW_LABEL_FILE}")
print("-" * 50)