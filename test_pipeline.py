import os
import cv2
import logging
from ultralytics import YOLO
from paddleocr import PaddleOCR

# 패들 로그 숨기기 (터미널 지저분해지는 것 방지)
logging.getLogger("ppocr").setLevel(logging.ERROR)

# ==========================================
# 1. 경로 세팅
# ==========================================
YOLO_MODEL_PATH = 'models/best.pt'
OCR_MODEL_PATH = 'models/inference_model_v3'
TEST_DIR = 'test_data_full_car'
SAVE_DIR = 'test_results_visual'

os.makedirs(SAVE_DIR, exist_ok=True)
print("🚀 [System] 여백(Padding) 알고리즘이 적용된 V3 테스트를 시작합니다...\n")

# ==========================================
# 2. 모델 초기화
# ==========================================
detector = YOLO(YOLO_MODEL_PATH)
ocr = PaddleOCR(
    det=False,
    rec_model_dir=OCR_MODEL_PATH, 
    lang='korean', # 한국어 강제 지정
    # 본인의 사전 경로가 맞는지 다시 한번 확인!
    rec_char_dict_path='C:/Users/user/AppData/Local/Programs/Python/Python39/lib/site-packages/paddleocr/ppocr/utils/dict/korean_dict.txt', 
    use_gpu=False,
    show_log=False # 다운로드 등 지저분한 로그 출력 차단
)

# ==========================================
# 3. 자동 채점 & 시각화 저장 로직
# ==========================================
correct_count = 0
total_count = 0

print("-" * 50)
print(f"{'파일명(정답)':<15} | {'V3 예측 결과':<15} | {'상태':<5} | {'확신도'}")
print("-" * 50)

for filename in os.listdir(TEST_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    total_count += 1
    img_path = os.path.join(TEST_DIR, filename)
    true_label = os.path.splitext(filename)[0] 
    
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w, _ = img.shape # 원본 이미지 크기

    # [Stage 1] YOLO 탐지
    results = detector(img, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print(f"{true_label:<15} | {'-':<15} | {'Fail':<5} | YOLO 실패")
        continue

    box = boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    # 💡 [핵심] 상하좌우 여백(Margin) 7% 추가 알고리즘
    box_width = x2 - x1
    box_height = y2 - y1
    
    margin_x = int(box_width * 0.07)
    margin_y = int(box_height * 0.07)

    # 이미지 범위를 벗어나지 않도록 안전하게 여백 확보
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    cropped_plate = img[y1:y2, x1:x2]

    # ==========================================
    # 💡 [핵심 추가] OCR 전용 이미지 시력 교정 (Pre-processing)
    # ==========================================
    # 1. 픽셀 보간법으로 해상도 2배 강제 확대 (흐릿한 글자 복원)
    cropped_plate = cv2.resize(cropped_plate, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 2. 이미지를 흑백으로 변환
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE 알고리즘 적용: 빛 반사나 그림자를 제거하고 글자와 배경의 대비를 극대화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 4. 패들OCR은 3채널(컬러)을 요구하므로 다시 차원 복구
    cropped_plate = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    # ==========================================

    # [Stage 2] V3 모델 인식
    ocr_result = ocr.ocr(cropped_plate, det=False, cls=False)

    if not ocr_result or not ocr_result[0]:
        print(f"{true_label:<15} | {'-':<15} | {'Fail':<5} | V3 실패")
        continue

    pred_text = ocr_result[0][0][0]
    confidence = ocr_result[0][0][1]
    pred_text_clean = pred_text.replace(" ", "")

    if pred_text_clean == true_label:
        correct_count += 1
        status = "✅ O"
    else:
        status = "❌ X"

    print(f"{true_label:<15} | {pred_text_clean:<15} | {status:<5} | {confidence:.2f}")

    mark = "O" if status == "✅ O" else "X"
    save_filename = f"[{mark}]_GT_{true_label}_PRED_{pred_text_clean}.jpg"
    cv2.imwrite(os.path.join(SAVE_DIR, save_filename), cropped_plate)

# ==========================================
# 4. 최종 리포트
# ==========================================
if total_count > 0:
    accuracy = (correct_count / total_count) * 100
    print("-" * 50)
    print(f"🎯 [최종 리포트] V3 모델 실전 테스트 결과")
    print(f"▶ 최종 정확도: {accuracy:.2f}% ({correct_count}/{total_count})")
    print("-" * 50)