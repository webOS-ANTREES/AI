import torch
import os
from pathlib import Path
from PIL import Image

# 학습된 모델 로드
model_path = 'C:/Users/antl/yolov5/runs/train/exp6/weights/best.pt'
model = torch.hub.load('C:/Users/antl/yolov5', 'custom', path=model_path, source='local')

# 폴더 내 모든 이미지를 테스트
folder_path = 'C:/Users/antl/SW_22th_contest/Object_Detective_Data/raw_data'
images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg'))]

# 결과 저장할 폴더
save_path = 'C:/Users/antl/SW_22th_contest/Object_Detective_Data/detected_strawberries'

# 결과 폴더가 없다면 생성
Path(save_path).mkdir(parents=True, exist_ok=True)

for img_path in images:
    # 추론 실행
    results = model(img_path)
    results.render()  # 결과 이미지에 탐지 박스와 라벨을 렌더링

    # 탐지된 객체들의 이미지를 저장
    for i, det in enumerate(results.xyxy[0]):  # 결과의 바운딩 박스를 순회
        if det[5] == 1:  # 클래스 1 (딸기) 인 경우만 처리
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            img = Image.open(img_path).crop((x1, y1, x2, y2))
            save_file_path = os.path.join(save_path, f"strawberry_{i}_{os.path.basename(img_path)}")
            img.save(save_file_path)
            print(f"Saved strawberry image to {save_file_path}")  # 저장 경로 로그 출력

print("All detected strawberries have been processed and saved.")
