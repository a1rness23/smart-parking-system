import os

txt_file = 'train_data_draft.txt'
img_dir = 'rec_train_images'

existing_files = set()
fixed_lines = []

# 1. 기존 텍스트 파일 읽어서 띄어쓰기 오류를 탭(\t)으로 완벽 교정
with open(txt_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if not line: continue
    
    # 빈칸이나 탭 기준으로 분리 (파일명엔 띄어쓰기가 없으므로 안전함)
    parts = line.split()
    filename = parts[0]
    label = parts[1] if len(parts) > 1 else ""

    existing_files.add(filename)
    # 무조건 탭(\t)으로 다시 조립
    fixed_lines.append(f"{filename}\t{label}\n")

# 2. rec_train_images 폴더를 뒤져서 텍스트 파일에 없는 사진(누락본) 찾기
missing_files = []
if os.path.exists(img_dir):
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if filename not in existing_files:
                missing_files.append(filename)

# 3. 누락된 파일들을 텍스트 파일 맨 밑에 대기 상태로 추가
for missing in missing_files:
    fixed_lines.append(f"{missing}\t\n")

# 4. 깔끔하게 정돈된 내용으로 원본 파일 덮어쓰기
with open(txt_file, 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("="*50)
print("🎉 [해결사 로직 완료]")
print("1️⃣ 모든 띄어쓰기 오류가 탭(\\t)으로 완벽 교정되었습니다.")
print(f"2️⃣ 폴더에만 있던 누락된 파일 {len(missing_files)}개가 텍스트 파일 맨 아래에 추가되었습니다.")
print("="*50)