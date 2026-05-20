import os
from ultralytics import YOLO

model = YOLO("models/char_yolo_best.pt")
test_folder = "test_data_cropped_plates"

# 💡 핵심 1: Roboflow 데이터셋의 영어 라벨을 한글로 바꿔주는 마법의 사전
label_map = {
    'ga': '가', 'na': '나', 'da': '다', 'ra': '라', 'ma': '마', 'ba': '바', 'sa': '사', 'a': '아', 'ja': '자',
    'geo': '거', 'neo': '너', 'deo': '더', 'reo': '러', 'meo': '머', 'beo': '버', 'seo': '서', 'eo': '어', 'jeo': '저',
    'go': '고', 'no': '노', 'do': '도', 'ro': '로', 'mo': '모', 'bo': '보', 'so': '소', 'o': '오', 'jo': '조',
    'gu': '구', 'nu': '누', 'du': '두', 'ru': '루', 'mu': '무', 'bu': '부', 'su': '수', 'u': '우', 'ju': '주',
    'ha': '하', 'heo': '허', 'ho': '호'
}

print("🔍 YOLO 글자 단위 인식 대량 테스트 시작...\n")

correct_count = 0
total_count = 0

for filename in os.listdir(test_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        total_count += 1
        img_path = os.path.join(test_folder, filename)
        true_label = os.path.splitext(filename)[0].replace(" ", "")

        # 💡 약간 흐릿한 글자도 잘 잡을 수 있도록 conf=0.15 (확신도 15% 이상이면 모두 인식) 추가
        results = model(img_path, verbose=False, conf=0.25, iou=0.4)

        detected_chars = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            class_name = model.names[cls_id]
            
            if class_name == "license_plate" or class_name == "car":
                continue
                
            korean_char = label_map.get(class_name, class_name)
            x1 = box.xyxy[0][0].item()
            detected_chars.append({"char": korean_char, "x1": x1})
            
        detected_chars = sorted(detected_chars, key=lambda k: k['x1'])
        pred_label = "".join([item['char'] for item in detected_chars])
        
        if pred_label == true_label:
            correct_count += 1
            match_result = "✅ 일치"
        else:
            match_result = "❌ 불일치"
            
        print(f"[{match_result}] 정답: {true_label:8} | 모델 예측: {pred_label}")

accuracy = (correct_count / total_count) * 100
print("\n" + "=" * 40)
print(f"🏆 총 {total_count}장 중 {correct_count}장 성공!")
print(f"📈 최종 인식 정확도: {accuracy:.1f}%")
print("=" * 40)