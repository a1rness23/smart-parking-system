import os

txt_file = 'train_data_draft.txt'
img_dir = 'rec_train_images'

# 1. 텍스트 파일 내용 읽기 및 탭(\t) 강제 정규화
with open(txt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

txt_filenames = set()
normalized_lines = []

for line in lines:
    line = line.strip()
    if not line: continue
    
    # 공백(스페이스/탭)을 기준으로 완벽하게 두 덩어리로 분리
    parts = line.split() 
    if len(parts) >= 2:
        filename = parts[0]
        label = parts[1]
        txt_filenames.add(filename)
        # 💡 무조건 '1개의 탭'으로 다시 조립해서 저장!
        normalized_lines.append(f"{filename}\t{label}\n")

# 정규화된 내용으로 원본 파일 덮어쓰기
with open(txt_file, 'w', encoding='utf-8') as f:
    f.writelines(normalized_lines)

# 2. 폴더 내 실제 이미지 파일 목록 가져오기
actual_images = set()
if os.path.exists(img_dir):
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            actual_images.add(f)

# 3. 대조 작업 (Set의 차집합 활용)
missing_in_folder = txt_filenames - actual_images  
missing_in_txt = actual_images - txt_filenames     

print("="*50)
print("✅ [STEP 1] 탭(\\t) 정규화 완료!")
print("스페이스나 불규칙한 공백이 모두 1개의 탭으로 완벽하게 변환되었습니다.")
print("="*50)

print(f"⚠️ [STEP 2] 텍스트 파일엔 적혀있는데, 폴더엔 없는 사진 (총 {len(missing_in_folder)}개)")
for f in missing_in_folder:
    print(f" - {f}")
    
print("-" * 50)
print(f"⚠️ [STEP 3] 폴더엔 있는데, 텍스트 파일엔 없는 사진 (총 {len(missing_in_txt)}개)")
for f in missing_in_txt:
    print(f" - {f}")
print("="*50)