import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# ⚙️ 1. 기본 설정
# ==========================================
OUTPUT_DIR = "./new_synthetic_plates_v2"
LABEL_FILE = "./new_train_v2.txt"
NUM_IMAGES = 10000

# 💡 대한민국 실제 허용 번호판 한글 (총 40개 완벽 반영)
# 가~마, 거~머, 고~모, 구~무, 버~주 + 렌터카(하허호) + 영업용(아바사자) + 택배(배)
VALID_CHARS = list("가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호아바사자배")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

try:
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 60)
except:
    print("❌ 폰트를 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# ==========================================
# 🧠 2. 번호판 텍스트 및 실전 노이즈 생성 로직
# ==========================================
def generate_plate_string():
    prefix = str(random.randint(10, 999))
    mid = random.choice(VALID_CHARS)
    suffix = f"{random.randint(1, 9999):04d}"
    return f"{prefix}{mid}{suffix}"

def add_real_world_noise(image):
    np_img = np.array(image)

    # 1. 흑백 카메라 노이즈 (ISO 노이즈) - 알록달록한 점 제거!
    row, col, ch = np_img.shape
    sigma = random.uniform(5, 15) # 노이즈 강도
    # 흑백(1채널) 노이즈를 만들어서 RGB 3채널에 동일하게 복사
    gauss = np.random.normal(0, sigma, (row, col, 1)).astype(np.float32)
    gauss = np.repeat(gauss, 3, axis=2) 
    
    np_img = np_img.astype(np.float32) + gauss
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)

    # 2. 다양한 블러 적용 (가우시안 블러 vs 움직이는 차량의 모션 블러)
    if random.random() > 0.5:
        # 가우시안 블러 (초점 흐림)
        blur_radius = random.uniform(0.5, 1.5)
        np_img = cv2.GaussianBlur(np_img, (0, 0), blur_radius)
    else:
        # 모션 블러 (차량이 움직일 때 찍힌 흔들림)
        kernel_size = random.choice([3, 5])
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        np_img = cv2.filter2D(np_img, -1, kernel_motion_blur)

    # 3. 주차장 환경의 불균일한 조명 (그림자 그라데이션)
    if random.random() > 0.4:
        alpha = random.uniform(0.5, 0.8) # 그림자의 어두운 정도
        gradient = np.linspace(alpha, 1.0, col).reshape(1, col, 1)
        if random.random() > 0.5:
            gradient = np.flip(gradient, axis=1) # 좌/우 방향 랜덤
        np_img = (np_img * gradient).astype(np.uint8)

    # 4. 카메라 설치 각도에 따른 삐딱함 (회전)
    image = Image.fromarray(np_img)
    image = image.rotate(random.uniform(-3.0, 3.0), expand=False, fillcolor=(255, 255, 255))
    
    return image

# ==========================================
# 🚀 3. 데이터 생성 실행
# ==========================================
print(f"⏳ 실전 노이즈가 적용된 완벽한 데이터 {NUM_IMAGES}장 생성을 시작합니다...")

with open(LABEL_FILE, 'w', encoding='utf-8') as f:
    for i in range(NUM_IMAGES):
        plate_text = generate_plate_string()
        
        img = Image.new('RGB', (300, 80), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 글자 그리기
        draw.text((20, 0), plate_text, font=font, fill=(0, 0, 0))
        
        # 업그레이드된 복합 노이즈 적용!
        final_img = add_real_world_noise(img)
        
        filename = f"plate_{i:05d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        final_img.save(filepath, quality=random.randint(70, 90)) # JPG 압축 손실 랜덤화
        
        f.write(f"{filename}\t{plate_text}\n")
        
        if (i + 1) % 1000 == 0:
            print(f"✅ {i + 1}장 생성 완료...")

print(f"🎉 V2 데이터 생성 완료! 폴더: {OUTPUT_DIR}, 정답지: {LABEL_FILE}")