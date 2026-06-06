import os

# 디버깅: 파일이 왜 안 찾아지는지 확인
img_dir = 'rec_train_images'
label_file = 'train_data_draft.txt'

with open(label_file, 'r', encoding='utf-8') as f:
    sample_line = f.readline().split(' ')[0]
    print(f"텍스트 파일 첫 번째 파일명: {sample_line}")
    print(f"존재 여부: {os.path.exists(os.path.join(img_dir, sample_line))}")
    print(f"폴더 안의 첫 번째 파일명: {os.listdir(img_dir)[0]}")