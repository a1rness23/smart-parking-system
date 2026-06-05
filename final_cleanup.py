import os

txt_file = 'train_data_draft.txt'
img_dir = 'rec_train_images'

seen_labels = set()
final_lines = []
removed_count = 0

with open(txt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line:
        continue
    
    # 💡 1. 띄어쓰기든 탭이든 무조건 분리해서 가져오기
    parts = line.split()
    if len(parts) < 2:
        continue 
        
    filename = parts[0]
    label = parts[1]
    
    # 💡 2. 중복 번호판 검사 (이미 텍스트가 같은 차가 있다면?)
    if label in seen_labels:
        removed_count += 1
        print(f"🗑️ 중복 데이터 삭제됨: {filename} (번호: {label})")
        
        # 텍스트에서 뺄 뿐만 아니라, 폴더에서도 과감하게 사진 삭제!
        img_path = os.path.join(img_dir, filename)
        if os.path.exists(img_path):
            os.remove(img_path)
        continue
        
    seen_labels.add(label)
    
    # 💡 3. 무조건 탭(\t) 1개로 재조립하여 안전한 리스트에 보관
    final_lines.append(f"{filename}\t{label}\n")

# 완벽해진 리스트로 텍스트 파일 덮어쓰기
with open(txt_file, 'w', encoding='utf-8') as f:
    f.writelines(final_lines)

print("="*50)
print("🎉 [최종 데이터셋 클렌징 완료]")
print(f"✅ 살아남은 순도 100% 실전 데이터: {len(final_lines)}개")
print(f"♻️ 내용이 겹쳐서 삭제된 중복 데이터: {removed_count}개")
print("✨ 띄어쓰기 오류 없이 모든 줄이 완벽한 탭(\\t)으로 교정되었습니다.")
print("="*50)