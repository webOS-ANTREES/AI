"""
 * Project : 제 22회 임베디드 소프트웨어 경진대회 - 딸기 열매 객체 탐색 알고리즘 구현 및 딸기 열매 익음 정도 판단 구현
 * Program Purpose and Features :
 * - load testset and Predict labels with Yolov5
 * Author : HG Kim
 * First Write Date : 2024.09.03
 * ============================================================
 * Program history
 * ============================================================
 * Author    		Date		    Version		        History
    HG Kim          2024.09.03      Model_with_Yolo.v1  모델 생성해봄.
    HG Kim          2024.09.18      Model_with_Yolo.v2  roboflow를 통한 모델 성능 향상
    HG Kim          2024.09.19      Model_with_Yolo.v3  dataset 업데이트
    HG Kim          2024.09.19      Model_with_Yolo.v4  yolov5를 통해서 딸기 객체 탐색과 열매 익음정도를 같이 학습시킨 후 해당 모델을 통해 한번에 판단.(익음,안익음 판단 GOOD, But 잎과 줄기를 딸기로 인식하는 확률 올라감)
    HG Kim          2024.09.20      Model_with_Yolo.v5  객체 탐색한 딸기의 캡처본 저장하는 함수 추가
"""
import torch
import os
from pathlib import Path
import cv2
import shutil

def load_model(model_path, model_source='local'):
    """
    YOLOv5 모델을 불러옵니다.
    :param model_path: 학습된 모델의 경로.
    :param model_source: 모델의 출처 (기본값은 'local').
    :return: 불러온 YOLOv5 모델.
    """
    return torch.hub.load('C:/Users/antl/yolov5', 'custom', path=model_path, source=model_source)

def save_detection_labels(detections, img_path, labels_save_path):
    """
    탐지된 객체에 대한 라벨(바운딩 박스 좌표, 클래스, 신뢰도)을 텍스트 파일로 저장합니다.
    :param detections: 모델로부터 탐지된 객체들.
    :param img_path: 처리된 이미지의 경로.
    :param labels_save_path: 탐지된 라벨 파일을 저장할 경로.
    """
    label_file_path = os.path.join(labels_save_path, os.path.basename(img_path).replace('.jpg', '.txt').replace('.png', '.txt'))
    
    with open(label_file_path, 'w') as label_file:
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection  # 바운딩 박스 좌표, 신뢰도, 클래스
            # YOLO 형식으로 저장: 클래스 중심 좌표 x_center, y_center 너비 width, 높이 height, 신뢰도 confidence
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            x_center = x1 + bbox_width / 2
            y_center = y1 + bbox_height / 2
            
            label_file.write(f"{int(cls)} {x_center:.4f} {y_center:.4f} {bbox_width:.4f} {bbox_height:.4f} {conf:.2f}\n")
    
    print(f"라벨이 {label_file_path}에 저장되었습니다.")

def process_images(model, folder_path, save_path, classes_to_filter=(0, 1)):
    """
    폴더에 있는 모든 이미지에서 객체 탐지를 실행하고, 결과를 바운딩 박스와 함께 저장하며, 탐지 정보를 텍스트 파일에 저장합니다.
    또한 원본 이미지를 저장합니다.
    :param model: YOLOv5 모델.
    :param folder_path: 이미지가 저장된 폴더의 경로.
    :param save_path: 결과가 저장될 폴더의 경로.
    :param classes_to_filter: 필터링할 클래스 ID들의 튜플 (기본값은 (0, 1)로, 익은 딸기와 안 익은 딸기를 의미).
    """
    # 이미지를 저장할 경로, 라벨을 저장할 경로, 원본 데이터를 저장할 경로 정의
    images_save_path = os.path.join(save_path, 'images')
    labels_save_path = os.path.join(save_path, 'labels')
    raw_data_save_path = os.path.join(save_path, 'raw_data')

    # 저장 디렉토리가 존재하지 않으면 생성
    Path(images_save_path).mkdir(parents=True, exist_ok=True)
    Path(labels_save_path).mkdir(parents=True, exist_ok=True)
    Path(raw_data_save_path).mkdir(parents=True, exist_ok=True)

    # 폴더에서 모든 이미지 파일 가져오기
    images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg'))]

    for img_path in images:
        # 추론 수행
        results = model(img_path)
        
        # 탐지 결과 추출
        detections = results.xyxy[0]
        
        # 클래스에 기반한 탐지 필터링 (0 - 익은 딸기, 1 - 안 익은 딸기)
        filtered_detections = detections[(detections[:, 5] == 0) | (detections[:, 5] == 1)]
        
        # 원본 이미지 로드
        img = cv2.imread(img_path)
        
        # 필터링된 객체에 바운딩 박스 그리기
        for detection in filtered_detections:
            x1, y1, x2, y2, conf, cls = detection  # 바운딩 박스 좌표, 신뢰도, 클래스
            
            # 클래스에 따라 색상과 라벨 설정
            if int(cls) == 0:
                color = (255, 0, 0)  # 익은 딸기는 초록색
                label = f'익은 딸기: {conf:.2f}'
            elif int(cls) == 1:
                color = (0, 0, 255)  # 안 익은 딸기는 빨간색
                label = f'안 익은 딸기: {conf:.2f}'
            
            # 바운딩 박스와 라벨 그리기
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 결과 이미지를 'images' 폴더에 저장
        save_file_path = os.path.join(images_save_path, os.path.basename(img_path))
        cv2.imwrite(save_file_path, img)

        # 탐지된 라벨을 텍스트 파일로 'labels' 폴더에 저장
        save_detection_labels(filtered_detections, img_path, labels_save_path)

        # 원본 이미지를 'raw_data' 폴더에 저장
        raw_image_save_path = os.path.join(raw_data_save_path, os.path.basename(img_path))
        shutil.copy(img_path, raw_image_save_path)  # 원본 이미지 복사
        print(f"원본 이미지가 {raw_image_save_path}에 저장되었습니다.")

        # 처리 상태 출력
        print(f"{img_path} 처리가 완료되어 {save_file_path}에 저장되었습니다.")

    print("모든 이미지, 라벨, 원본 데이터가 처리되어 저장되었습니다.")


    
def capture_bounding_boxes(Object_Detection_Result_Folder):
    """
    라벨 파일을 읽고, 해당 바운딩 박스 좌표에 맞춰 이미지를 잘라내어 익은 딸기와 안 익은 딸기로 분류하여 저장합니다.
    :param Object_Detection_Result_Folder: 라벨과 이미지를 포함한 폴더 경로.
    """
    
    images_folder = os.path.join(Object_Detection_Result_Folder, 'raw_data')
    labels_folder = os.path.join(Object_Detection_Result_Folder, 'labels')
    ripe_folder = os.path.join(Object_Detection_Result_Folder, 'captured_ripe_berry')
    unripe_folder = os.path.join(Object_Detection_Result_Folder, 'captured_unripe_berry')

    # 폴더가 없으면 생성
    os.makedirs(ripe_folder, exist_ok=True)
    os.makedirs(unripe_folder, exist_ok=True)

    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):
            # 라벨 파일 경로
            label_path = os.path.join(labels_folder, label_file)

            # 이미지 파일 이름 가져오기 (라벨 파일과 동일한 이름의 PNG 이미지로 가정)
            image_file = label_file.replace('.txt', '.png')
            image_path = os.path.join(images_folder, image_file)

            # 이미지 읽기
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: {image_path} 이미지를 불러올 수 없습니다.")
                continue

            # 이미지 크기 가져오기
            img_height, img_width, _ = image.shape
            print(f"이미지 크기: {img_width}x{img_height}")

            # 라벨 파일 읽기
            with open(label_path, 'r') as file:
                lines = file.readlines()
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    label_class = int(parts[0])  # 클래스 값 (0 또는 1)
                    x_center = float(parts[1])
                    y_center = float(parts[2]) 
                    box_width = float(parts[3])
                    box_height = float(parts[4]) 

                    # 바운딩 박스 좌표 계산
                    x_min = int(x_center - box_width / 2)
                    y_min = int(y_center - box_height / 2)
                    x_max = int(x_center + box_width / 2)
                    y_max = int(y_center + box_height / 2)

                    # 바운딩 박스 좌표와 크기 확인
                    print(f"바운딩 박스 좌표: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                    print(f"바운딩 박스 크기: 너비={x_max - x_min}, 높이={y_max - y_min}")

                    # 바운딩 박스가 이미지 범위를 벗어나지 않도록 제한
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width, x_max)
                    y_max = min(img_height, y_max)

                    # 자를 영역의 크기가 유효한지 확인
                    if x_max - x_min > 0 and y_max - y_min > 0:
                        # 바운딩 박스 영역 잘라내기
                        cropped_image = image[y_min:y_max, x_min:x_max]

                        # 저장 경로 설정
                        if label_class == 0:
                            output_folder = ripe_folder
                        else:
                            output_folder = unripe_folder

                        # 잘라낸 이미지를 저장 (파일명은 원본 이미지 이름 + 번호로 저장)
                        output_path = os.path.join(output_folder, f"{label_file.replace('.txt', '')}_crop_{idx}.png")
                        cv2.imwrite(output_path, cropped_image)
                        print(f"잘라낸 이미지 저장됨: {output_path}")
                    else:
                        print(f"Warning: 너무 작은 바운딩 박스 영역. 라벨 파일: {label_file}, 좌표: {(x_min, y_min, x_max, y_max)}")

# Example of how to call the functions
if __name__ == "__main__":
    model_path = 'C:/Users/antl/yolov5/runs/train/exp11/weights/best.pt'
    folder_path = 'C:/Users/antl/SW_22th_contest/Object_Detective_Data/raw_data'
    save_path = 'C:/Users/antl/SW_22th_contest/Object_Detective_Data/results_v4'

    model = load_model(model_path)
    process_images(model, folder_path, save_path)
    capture_bounding_boxes(save_path)
