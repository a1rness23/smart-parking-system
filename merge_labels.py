import os

train_label_file = 'train_data_draft.txt'
test_folder = 'test_results_visual'
output_file = 'combined_labels.txt'

# 1. 기존 라벨 읽기
combined_data = []
with open(train_label_file, 'r', encoding='utf-8') as f:
    for line in f:
        combined_data.append(line.strip())

# 2. 새로운 테스트 데이터 추가 (라벨은 일단 파일명과 같게 설정)
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        # 일단 파일명을 정답라벨로 넣음 (이후 텍스트 파일에서 수정 필요)
        label = filename.split('_')[0] 
        combined_data.append(f"{filename} {label}")

# 3. 저장
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(combined_data))

print(f"완료! {output_file} 파일을 열어서 내용을 확인하고, 오타가 있는 라벨만 수정하세요.")