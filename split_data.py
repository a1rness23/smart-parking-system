import random

# 1. 뻥튀기된 전체 라벨 읽기
with open('aug_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 2. 순서가 한쪽으로 쏠리지 않게 완벽하게 섞기
random.shuffle(lines)

# 3. 9:1 비율로 쪼개기 (Train 90%, Valid 10%)
split_index = int(len(lines) * 0.9)
train_lines = lines[:split_index]
val_lines = lines[split_index:]

# 4. 파일로 저장 (버전 헷갈리지 않게 v5로 명명)
with open('train_v5.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open('val_v5.txt', 'w', encoding='utf-8') as f:
    f.writelines(val_lines)

print("=" * 50)
print(f"✅ 정답지 분리 완료!")
print(f"▶ 학습용(Train): {len(train_lines)}장")
print(f"▶ 모의고사용(Valid): {len(val_lines)}장")
print("=" * 50)