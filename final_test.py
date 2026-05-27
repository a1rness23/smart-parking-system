import os
import re
from paddleocr import PaddleOCR

# ==========================================
# ⚙️ 1. 경로 설정 및 AI 모델 초기화
# ==========================================
CUSTOM_MODEL_DIR = "./models/inference_model_v2"
CROPPED_DIR = "./test_data_cropped_plates"

ocr = PaddleOCR(
    # 💡 파일들(inference.pdmodel 등)이 직접 들어있는 폴더 경로를 정확히 적어주세요!
    # (만약 inference 하위 폴더 안에 있다면 경로 끝에 /inference 를 붙여주세요)
    rec_model_dir="./models/inference_model_v4_real_stable", 
    
    lang='korean',
    use_gpu=False,
    show_log=False,  
    rec=True,
    det=False,
    cls=False
)

def evaluate_accuracy_rec_only(folder_path):
    print("="*70)
    print("📊 [순수 인식(Rec) 채점] 탐지(Det)를 끄고 평가합니다.")
    print("="*70)
    
    if not os.path.exists(folder_path):
        print("❌ 폴더를 찾을 수 없습니다.")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_count = len(image_files)
    correct_count = 0

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        
        # 1. 정답 추출
        true_label = os.path.splitext(img_name)[0]
        true_label = re.sub(r'[^가-힣0-9]', '', true_label)
        
        # 💡 2. AI 판독: det=False 옵션으로 '탐지'를 끄고 '인식'만 수행!
        # (잘린 이미지를 통째로 한 줄의 텍스트로 간주함)
        result = ocr.ocr(img_path, det=False, cls=False)
        
        # 3. 결과 파싱 (det=False일 때는 결과 구조가 [[('글자', 확률)]] 형태임)
        predicted_text = ""
        if result and len(result) > 0 and result[0]:
            predicted_text = result[0][0][0] # 첫 번째 예측된 텍스트 가져오기
                
        # 4. 특수문자 제거 및 채점
        predicted_text = re.sub(r'[^가-힣0-9]', '', predicted_text)
        
        is_correct = (true_label == predicted_text)
        if is_correct:
            correct_count += 1
            match_str = "✅ 정답"
        else:
            match_str = "❌ 오답"
            
        print(f"파일명: {img_name:<15} | 정답: {true_label:<10} | AI 예측: {predicted_text:<10} | 결과: {match_str}")
        
    print("="*70)
    accuracy = (correct_count / total_count) * 100
    print(f"🏆 최종 성적: {correct_count}문제 정답 / 총 {total_count}문제 (정확도: {accuracy:.2f}%)")
    print("="*70)

# ==========================================
# 🏃‍♂️ 2. 채점 실행
# ==========================================
evaluate_accuracy_rec_only(CROPPED_DIR)