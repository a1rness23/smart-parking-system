import os
from paddleocr import PaddleOCR

# 1. OCR 엔진 로딩 (에포크 500 모델 폴더 유지)
ocr = PaddleOCR(det=False, rec_model_dir='models/my_final_v3', 
                rec_char_dict_path='models/korean_dict.txt', use_gpu=False, show_log=False)

# 2. 정답 파일 로드
label_map = {}
with open('train_data_draft.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            label_map[parts[0]] = parts[1]

# 3. 테스트 및 '번호판 전체 정확도' 계산
correct_plates = 0
total_plates = 0

print(f"{'파일명':<20} | {'GT':<10} | {'Pred':<10} | {'결과'}")
print("-" * 50)

for filename, gt in label_map.items():
    img_path = os.path.join('rec_train_images', filename)
    if not os.path.exists(img_path): continue
    
    res = ocr.ocr(img_path, det=False, cls=False)
    if res and res[0]:
        pred = res[0][0][0].replace("-", "").replace(" ", "")
        gt_clean = gt.replace("-", "").replace(" ", "")
        
        # 완전 일치 확인
        if pred == gt_clean:
            correct_plates += 1
            print(f"{filename[:15]:<20} | {gt_clean:<10} | {pred:<10} | ✅ OK")
        else:
            print(f"{filename[:15]:<20} | {gt_clean:<10} | {pred:<10} | ❌ Fail")
        
        total_plates += 1

if total_plates > 0:
    acc = (correct_plates / total_plates) * 100
    print(f"\n🎯 최종 번호판 인식 정확도(Exact Plate Match): {acc:.2f}% ({correct_plates}/{total_plates})")