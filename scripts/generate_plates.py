import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# 💡 실전 학습을 위한 10,000장 대량 생산! (1~2분 소요)
NUM_PLATES = 10000
OUTPUT_DIR = "synthetic_plates_v4"
LABEL_FILE = "train_v4.txt"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

chars_center = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후"
chars_region = ["서울", "경기", "인천", "부산", "대구", "대전", "광주", "울산", "제주"]

font_path = "C:/Windows/Fonts/malgunbd.ttf" 
font_large = ImageFont.truetype(font_path, 70)
font_medium = ImageFont.truetype(font_path, 60)
font_small = ImageFont.truetype(font_path, 45)

print(f"🚀 [V3] 노이즈가 추가된 실전용 가짜 번호판 {NUM_PLATES}장 생성을 시작합니다...")

with open(LABEL_FILE, "w", encoding="utf-8") as f:
    for i in range(NUM_PLATES):
        plate_type = random.choice(["NEW_WHITE", "COMMERCIAL_YELLOW", "OLD_GREEN"])
        
        # 1. 캔버스 가로 사이즈(width)를 30~50px 씩 더 늘려서 우측 여백 확보!
        if plate_type == "NEW_WHITE":
            bg_color = (255, 255, 255)
            text_color = (0, 0, 0)
            img_size = (480, 110) # 450 -> 480 확장
            front = str(random.randint(100, 999))
            mid = random.choice(chars_center)
            back = str(random.randint(1000, 9999))
            display_text = f"{front}{mid}   {back}" # 여백 조금 더 추가
            label_text = f"{front}{mid}{back}"
            
        elif plate_type == "COMMERCIAL_YELLOW":
            bg_color = (255, 204, 0)
            text_color = (0, 0, 0)
            img_size = (430, 110) # 400 -> 430 확장
            front = str(random.randint(10, 99))
            mid = random.choice(chars_center)
            back = str(random.randint(1000, 9999))
            display_text = f"{front}{mid}   {back}"
            label_text = f"{front}{mid}{back}"
            
        else:
            bg_color = (0, 153, 51)
            text_color = (255, 255, 255)
            img_size = (320, 160) # 300 -> 320 확장
            region = random.choice(chars_region)
            front = str(random.randint(10, 99))
            mid = random.choice(chars_center)
            back = str(random.randint(1000, 9999))
            display_text_top = f"{region} {front}"
            display_text_bottom = f"{mid}  {back}"
            label_text = f"{region}{front}{mid}{back}"

        # 2. 기본 이미지 그리기
        img = Image.new('RGB', img_size, color=bg_color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([5, 5, img_size[0]-5, img_size[1]-5], outline=text_color, width=3)
        
        if plate_type == "OLD_GREEN":
            # 시작 X 좌표도 살짝 우측으로 밀어 안정감 확보
            draw.text((65, 10), display_text_top, fill=text_color, font=font_small)
            draw.text((45, 70), display_text_bottom, fill=text_color, font=font_medium)
        else:
            draw.text((40, 15), display_text, fill=text_color, font=font_large)

        # --------------------------------------------------------
        # 💡 [핵심] 데이터 증강(Augmentation): AI를 강하게 키우는 시련 
        # --------------------------------------------------------
        
        # A. 살짝 기울이기 (카메라 각도 삐뚤어짐 모방: -3도 ~ 3도)
        angle = random.uniform(-3, 3)
        img = img.rotate(angle, fillcolor=bg_color, expand=False)
        
        # B. 밝기 조절 (밤낮, 그늘 모방: 60% ~ 120% 밝기)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.6, 1.2))
        
        # C. 가우시안 블러 (초점 나감, 비 오는 날 모방: 0 ~ 1.5 수준)
        # 랜덤하게 50% 확률로 흐리게 만듦
        # C. 가우시안 블러 (초점 나감, 비 오는 날 모방: 0 ~ 1.5 수준)
        # 랜덤하게 50% 확률로 흐리게 만듦
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # 👇👇👇 [여기에 V4 코드 추가] 👇👇👇
        # D. 강제 훼손 및 가림 (Occlusion & Scratches) - 30% 확률로 발생
        if random.random() > 0.7:
            draw_noise = ImageDraw.Draw(img)
            # 1. 스크래치 (검은색/흰색 얇은 선을 무작위로 1~3개 그음)
            for _ in range(random.randint(1, 3)):
                start_pt = (random.randint(0, img_size[0]), random.randint(0, img_size[1]))
                end_pt = (random.randint(0, img_size[0]), random.randint(0, img_size[1]))
                draw_noise.line([start_pt, end_pt], fill=random.choice([(0,0,0), (255,255,255)]), width=random.randint(1, 3))
            
            # 2. 진흙/폭설 가림 (번호판 위에 크고 작은 얼룩을 1~2개 찍음)
            for _ in range(random.randint(1, 2)):
                x, y = random.randint(10, img_size[0]-20), random.randint(10, img_size[1]-20)
                r = random.randint(5, 15) # 얼룩 크기
                draw_noise.ellipse([x-r, y-r, x+r, y+r], fill=random.choice([(200,200,200), (50,50,50)]))

        # --------------------------------------------------------
        
        # 3. 저장
        filename = f"plate_{i:05d}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        img.save(save_path)
        f.write(f"synthetic_plates_v3/{filename}\t{label_text}\n")

print(f"✅ V3 생성 완료! '{OUTPUT_DIR}' 폴더에 1만 장의 실전 데이터가 준비되었습니다.")