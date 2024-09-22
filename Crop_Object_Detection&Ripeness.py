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
    
"""
import torch
import os
from pathlib import Path
import cv2
import shutil

def load_model(model_path, model_source='local'):
    """
    Load a YOLOv5 model.
    :param model_path: Path to the trained model.
    :param model_source: Source of the model (default is 'local').
    :return: Loaded YOLOv5 model.
    """
    return torch.hub.load('C:/Users/antl/yolov5', 'custom', path=model_path, source=model_source)

def save_detection_labels(detections, img_path, labels_save_path):
    """
    Save detection labels (bounding box coordinates, class, and confidence) to a text file.
    :param detections: Detected objects from the model.
    :param img_path: Path of the processed image.
    :param labels_save_path: Path where the detection label file will be saved.
    """
    label_file_path = os.path.join(labels_save_path, os.path.basename(img_path).replace('.jpg', '.txt').replace('.png', '.txt'))
    
    with open(label_file_path, 'w') as label_file:
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection  # Bounding box coordinates, confidence, class
            # Write in YOLO format: class x_center y_center width height confidence
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            x_center = x1 + bbox_width / 2
            y_center = y1 + bbox_height / 2
            
            label_file.write(f"{int(cls)} {x_center:.4f} {y_center:.4f} {bbox_width:.4f} {bbox_height:.4f} {conf:.2f}\n")
    
    print(f"Labels saved to {label_file_path}")

def process_images(model, folder_path, save_path, classes_to_filter=(0, 1)):
    """
    Run object detection on all images in a folder, save results with bounding boxes, 
    and store detection information in text files.
    :param model: YOLOv5 model.
    :param folder_path: Path to the folder containing images.
    :param save_path: Path to the folder where results will be saved.
    :param classes_to_filter: Tuple of class IDs to filter (default is (0, 1) for ripe and unripe strawberries).
    """
    # Define the paths for saving images and labels
    images_save_path = os.path.join(save_path, 'images')
    labels_save_path = os.path.join(save_path, 'labels')

    # Create save directories if they don't exist
    Path(images_save_path).mkdir(parents=True, exist_ok=True)
    Path(labels_save_path).mkdir(parents=True, exist_ok=True)

    # Get all image files from folder
    images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg'))]

    for img_path in images:
        # Perform inference
        results = model(img_path)
        
        # Extract detection results
        detections = results.xyxy[0]
        
        # Filter detections based on class (0 - ripe, 1 - unripe)
        filtered_detections = detections[(detections[:, 5] == 0) | (detections[:, 5] == 1)]
        
        # Load original image
        img = cv2.imread(img_path)
        
        # Draw bounding boxes on filtered objects
        for detection in filtered_detections:
            x1, y1, x2, y2, conf, cls = detection  # Bounding box coordinates, confidence, class
            
            # Set color and label based on class
            if int(cls) == 0:
                color = (255, 0, 0)  # Green for ripe strawberry
                label = f'Ripe: {conf:.2f}'
            elif int(cls) == 1:
                color = (0, 0, 255)  # Red for unripe strawberry
                label = f'Unripe: {conf:.2f}'
            
            # Draw bounding box and label
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Save the result image in the 'images' folder
        save_file_path = os.path.join(images_save_path, os.path.basename(img_path))
        cv2.imwrite(save_file_path, img)

        # Save detection labels to a text file in the 'labels' folder
        save_detection_labels(filtered_detections, img_path, labels_save_path)

        # Print processing status
        print(f"Processed {img_path}, saved to {save_file_path}")

    print("All images and labels have been processed and saved.")
    
def capture_bounding_boxes(a_folder):
    images_folder = os.path.join(a_folder, 'images')
    labels_folder = os.path.join(a_folder, 'labels')
    ripe_folder = os.path.join(a_folder, 'captured_ripe_berry')
    unripe_folder = os.path.join(a_folder, 'captured_unripe_berry')

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
# 'a' 폴더 경로 예시
# Example of how to call the functions
if __name__ == "__main__":
    model_path = 'C:/Users/antl/yolov5/runs/train/exp11/weights/best.pt'
    folder_path = 'C:/Users/antl/SW_22th_contest/Object_Detective_Data/raw_data'
    save_path = 'C:/Users/antl/SW_22th_contest/Object_Detective_Data/results_v4'

    model = load_model(model_path)
    process_images(model, folder_path, save_path
    a_folder_path = 'C:/Users/antl/SW_22th_contest/Object_Detective_Data/results_v4'
    capture_bounding_boxes(a_folder_path)

