import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re

print("🤖 [안전 제일주의 AI] 글자 훼손 없는 안전 크롭 & 자동 라벨링을 시작합니다...")

# 💡 탐지(det) 해고! 오직 글자 읽기(rec) 기능만 씁니다.
ocr_model = PaddleOCR(
    rec_model_dir="./models/inference_model_v4_real_stable", 
    lang='korean', use_gpu=False, show_log=False,
    det=False, rec=True, cls=False, 
    ocr_version='PP-OCRv3', rec_image_shape="3, 48, 320"
)

unlabeled_dir = 'new_raw_images' 
output_img_dir = 'rec_train_images' 
output_file = 'train_data_draft.txt'

if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

success_count = 0
duplicate_count = 0
seen_plates = set()

with open(output_file, 'w', encoding='utf-8') as f:
    for filename in os.listdir(unlabeled_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue

        # 1. 작성자님 특허: 파일명에서 숫자 추출
        match = re.match(r'^(\d{2,3})-(\d{4})', filename)
        if not match: continue
        front_num, back_num = match.group(1), match.group(2)

        img_path = os.path.join(unlabeled_dir, filename)
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # ==========================================
        # 💡 [핵심] 안전 제일주의 수동 크롭
        # 좌우는 절대 안 건드림! 위아래 범퍼/나사만 8%씩 살짝 깎아냅니다.
        # ==========================================
        h, w = img.shape[:2]
        crop_y = int(h * 0.08) 
        
        safe_crop = img[crop_y:h-crop_y, 0:w]

        # 오려낸 안전한 사진 저장! (이게 진짜 훈련 데이터입니다)
        save_path = os.path.join(output_img_dir, filename)
        is_success, im_buf_arr = cv2.imencode(".jpg", safe_crop)
        if is_success:
            im_buf_arr.tofile(save_path)

        # 2. 한글 읽기를 돕기 위한 뻥튀기 (학습용 이미지는 원본 safe_crop으로 놔두고, 읽기용만 키움)
        eval_img = cv2.resize(safe_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # 3. 모델 판독 (det=False로 통째로 읽기)
        ocr_res = ocr_model.ocr(eval_img, det=False, cls=False)
        
        hangul_char = "?"
        if ocr_res and len(ocr_res) > 0 and ocr_res[0]:
            raw_text = ocr_res[0][0][0]
            hangul_match = re.search(r'([가-힣])', raw_text)
            if hangul_match:
                hangul_char = hangul_match.group(1)

        predicted_text = f"{front_num}{hangul_char}{back_num}"

        # 중복 검사
        if predicted_text in seen_plates and hangul_char != "?":
            print(f"♻️ 중복 패스: {filename:<15} ({predicted_text})")
            duplicate_count += 1
            if os.path.exists(save_path): os.remove(save_path)
            continue

        seen_plates.add(predicted_text)
        success_count += 1
        
        f.write(f"{filename}\t{predicted_text}\n")
        print(f"🎯 처리 완료: {filename:<20} ➡ {predicted_text}")

print("="*50)
print(f"🎉 안전 라벨링 완료! (저장: {success_count}장 / 중복 제외: {duplicate_count}장)")
print(f"1️⃣ 'rec_train_images' 폴더의 글자들이 100% 온전하게 살아있는지 확인해 보세요!")
print(f"2️⃣ '{output_file}'을 열어 '?'나 틀린 한글만 수정해 주세요.")
print("="*50)