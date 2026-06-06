import os
from paddleocr import PaddleOCR

# 1. OCR 엔진 로딩
ocr = PaddleOCR(det=False, rec_model_dir='models/my_final_v3', 
                rec_char_dict_path='models/korean_dict.txt', use_gpu=False, show_log=False)

# 2. 통합 라벨 파일 로드
label_map = {}
with open('combined_labels.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2: label_map[parts[0]] = parts[1]

# 3. 평가 시작
correct = 0
total = 0

print(f"{'파일명':<25} | {'결과'}")
print("-" * 40)

for filename, gt in label_map.items():
    # 폴더 두 곳을 모두 뒤져서 파일을 찾음
    img_path = None
    if os.path.exists(os.path.join('rec_train_images', filename)):
        img_path = os.path.join('rec_train_images', filename)
    elif os.path.exists(os.path.join('test_results_visual', filename)):
        img_path = os.path.join('test_results_visual', filename)
    
    if not img_path: continue
    
    res = ocr.ocr(img_path, det=False, cls=False)
    if res and res[0]:
        pred = res[0][0][0].replace("-", "").replace(" ", "")
        gt_clean = gt.replace("-", "").replace(" ", "")
        
        if pred == gt_clean:
            correct += 1
            print(f"{filename[:20]:<25} | ✅ OK")
        else:
            print(f"{filename[:20]:<25} | ❌ Fail (Pred: {pred})")
        total += 1

print(f"\n🎯 최종 통합 인식 정확도: {(correct/total)*100:.2f}% ({correct}/{total})")