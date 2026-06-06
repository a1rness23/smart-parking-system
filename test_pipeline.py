import os
import sys
import cv2
import logging
from ultralytics import YOLO
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.ERROR)

# ==========================================
# 1. 경로 세팅 (상대 경로로 깔끔하게 정리!)
# ==========================================
YOLO_MODEL_PATH = 'models/best.pt'
OCR_MODEL_PATH = 'models/my_final_v3'
DICT_PATH = 'models/korean_dict.txt' # 💡 코랩에서 다운받은 사전 파일을 models 폴더에 넣으세요!

# 💡 [철벽 방어막] 진짜 V3 모델이 없으면 다운로드 못하게 강제 종료!
model_check_path = os.path.join(OCR_MODEL_PATH, 'inference.pdmodel')
if not os.path.exists(model_check_path):
    print("=" * 60)
    print("🚨 [긴급 에러] 모델 파일을 찾을 수 없습니다!")
    print(f"▶ 파이썬이 찾고 있는 위치: {model_check_path}")
    print("▶ models 폴더 안에 my_final_v3 폴더가 제대로 있는지 확인해 주세요!")
    print("=" * 60)
    sys.exit()

TEST_DIR = 'test_data_full_car'
SAVE_DIR = 'test_results_visual'

os.makedirs(SAVE_DIR, exist_ok=True)
print("🚀 [System] 1차 노이즈 + 2차 현실 데이터로 무장한 V3 테스트를 시작합니다...\n")

# ==========================================
# 2. 모델 초기화
# ==========================================
detector = YOLO(YOLO_MODEL_PATH)
ocr = PaddleOCR(
    det=False,
    rec_model_dir=OCR_MODEL_PATH, 
    ocr_version='PP-OCRv3', 
    rec_image_shape="3, 48, 320", # 💡 V3 모델 성능을 100% 끌어내는 필수 규격!
    rec_char_dict_path=DICT_PATH, 
    use_gpu=False,
    show_log=False
)

# ==========================================
# 3. 자동 채점 & 시각화 저장 로직
# ==========================================
correct_count = 0
total_count = 0

print("-" * 55)
print(f"{'파일명(정답)':<15} | {'V3 예측 결과':<15} | {'상태':<5} | {'확신도'}")
print("-" * 55)

for filename in os.listdir(TEST_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    total_count += 1
    img_path = os.path.join(TEST_DIR, filename)
    true_label = os.path.splitext(filename)[0] 
    
    img = cv2.imread(img_path)
    if img is None:
        continue

    # [Stage 1] YOLO 탐지
    results = detector(img, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print(f"{true_label:<15} | {'-':<15} | {'Fail':<5} | YOLO 실패")
        continue

    box = boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    # 💡 [여백 제거 완료] 타이트하게 자르기
    cropped_plate = img[y1:y2, x1:x2]
    
    # 예외 처리: 박스가 너무 작게 잡혔을 경우 에러 방지
    if cropped_plate.size == 0:
        continue

    # 💡 [마일드한 시력 교정] 해상도 2배 확대 (CNN 피처맵 소실 방지)
    # cropped_plate = cv2.resize(cropped_plate, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # [Stage 2] V3 모델 인식 (중복되던 코드 한 줄로 정리!)
    ocr_result = ocr.ocr(cropped_plate, det=False, cls=False)

    if not ocr_result or not ocr_result[0]:
        print(f"{true_label:<15} | {'-':<15} | {'Fail':<5} | V3 읽기 실패")
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
    print("-" * 55)
    print(f"🎯 [최종 리포트] 다단계 학습 V3 모델 실전 테스트 결과")
    print(f"▶ 최종 정확도: {accuracy:.2f}% ({correct_count}/{total_count})")
    print("-" * 55)