import os
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re

# ==========================================
# 💡 번호판 문법 검사 및 자동 교정기
# ==========================================
def correct_plate_format(text):
    clean_text = re.sub(r'[^0-9가-힣]', '', text)
    match = re.search(r'(\d{2,3}[가-힣]\d{4})', clean_text)
    if match: return match.group(1)
    
    correction_map = {'4': '나', '0': '어', '1': '너', '2': '러', '5': '도', '8': '가', '3': '다'}
    if len(clean_text) == 8 and clean_text[:3].isdigit() and clean_text[4:].isdigit():
        wrong_char = clean_text[3]
        if wrong_char in correction_map:
            return clean_text[:3] + correction_map[wrong_char] + clean_text[4:]
    elif len(clean_text) == 7 and clean_text[:2].isdigit() and clean_text[3:].isdigit():
        wrong_char = clean_text[2]
        if wrong_char in correction_map:
            return clean_text[:2] + correction_map[wrong_char] + clean_text[3:]
    return clean_text

# ==========================================
# ⚙️ 모델 초기화
# ==========================================
print("🧠 [준비 중] YOLO 및 V4 파인튜닝 PaddleOCR을 불러옵니다...")
yolo_model = YOLO('./models/best.pt')

ocr_model = PaddleOCR(
    rec_model_dir="./models/inference_model_v4_real_stable", 
    lang='korean', 
    use_gpu=False,    
    show_log=False,
    # 💡 황금 조합 1: 다시 det=True를 사용하여 글자만 핀셋으로 집어내게 합니다!
    det=True,         
    rec=True,
    cls=False
)

test_dir = 'test_data_full_car' 
total_images = 0
correct_predictions = 0

print("\n🚀 [정확도 100% 도전 (순백의 도화지 + 핀셋 탐지 전략)]")
print("-" * 50)

for filename in os.listdir(test_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
        continue

    total_images += 1
    true_label = os.path.splitext(filename)[0]
    true_label = re.sub(r'[^가-힣0-9]', '', true_label) 

    img_path = os.path.join(test_dir, filename)
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 1. YOLO로 번호판 찾기
    results = yolo_model(img, verbose=False)
    predicted_text = ""

    if len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # 박스를 여유 있게 자릅니다 (5% 여유)
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * 0.05)
        pad_y = int(box_height * 0.05)
        
        y1_p = max(0, y1 - pad_y)
        y2_p = min(img.shape[0], y2 + pad_y)
        x1_p = max(0, x1 - pad_x)
        x2_p = min(img.shape[1], x2 + pad_x)
        cropped = img[y1_p:y2_p, x1_p:x2_p]

        # 💡 황금 조합 2: 테두리 복제가 아니라, '깨끗한 하얀색(255) 도화지'를 사방에 덧대어 줍니다!
        # 글자가 가장자리에 있어도 절대 무시되지 않고 완벽하게 탐지됩니다.
        padded = cv2.copyMakeBorder(cropped, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # 💡 황금 조합 3: 2배 확대!
        final_img = cv2.resize(padded, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # 2. 파인튜닝 모델로 판독
        ocr_res = ocr_model.ocr(final_img, cls=False)

        if ocr_res and len(ocr_res) > 0 and ocr_res[0]:
            # det=True일 때의 결과 합치기
            sorted_res = sorted(ocr_res[0], key=lambda x: x[0][0][0])
            raw_text = "".join([line[1][0] for line in sorted_res])
            predicted_text = correct_plate_format(raw_text)

    # 3. 채점
    if predicted_text == true_label:
        correct_predictions += 1
        print(f"✅ [정답] 파일: {true_label:<10} ➡ 예측: {predicted_text:<10}")
    else:
        print(f"❌ [오답] 파일: {true_label:<10} ➡ 예측: {predicted_text:<10}")

print("-" * 50)
if total_images > 0:
    accuracy = (correct_predictions / total_images) * 100
    print(f"📊 [최종 결과] 총 {total_images}장 중 {correct_predictions}장 정답! (정확도: {accuracy:.2f}%)")
else:
    print("⚠️ 폴더에 테스트할 사진이 없습니다!")