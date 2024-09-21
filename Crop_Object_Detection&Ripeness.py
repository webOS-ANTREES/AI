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

def load_model(model_path, model_source='local'):
    """
    Load a YOLOv5 model.
    :param model_path: Path to the trained model.
    :param model_source: Source of the model (default is 'local').
    :return: Loaded YOLOv5 model.
    """
    return torch.hub.load('C:/Users/antl/yolov5', 'custom', path=model_path, source=model_source)

def process_images(model, folder_path, save_path, classes_to_filter=(0, 1)):
    """
    Run object detection on all images in a folder and save results with bounding boxes.
    :param model: YOLOv5 model.
    :param folder_path: Path to the folder containing images.
    :param save_path: Path to the folder where results will be saved.
    :param classes_to_filter: Tuple of class IDs to filter (default is (0, 1) for ripe and unripe strawberries).
    """
    # Create save directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)

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
                color = (0, 255, 0)  # Green for ripe strawberry
                label = f'Ripe: {conf:.2f}'
            elif int(cls) == 1:
                color = (0, 0, 255)  # Red for unripe strawberry
                label = f'Unripe: {conf:.2f}'
            
            # Draw bounding box and label
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Save the result image
        save_file_path = os.path.join(save_path, os.path.basename(img_path))
        cv2.imwrite(save_file_path, img)

        # Print processing status
        print(f"Processed {img_path}, saved to {save_file_path}")

    print("All images have been processed and saved.")

# Example of how to call the functions
if __name__ == "__main__":
    model_path = 'C:/Users/antl/yolov5/runs/train/exp11/weights/best.pt'
    folder_path = 'C:/Users/antl/SW_22th_contest/Object_Detective_Data/raw_data'
    save_path = 'C:/Users/antl/SW_22th_contest/Object_Detective_Data/results_v4/test'

    model = load_model(model_path)
    process_images(model, folder_path, save_path)
