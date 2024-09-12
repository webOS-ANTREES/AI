import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 데이터셋 경로 설정
train_dir = 'C:/Users/antl/Desktop/berry riped data/test'  # 학습용 데이터 경로 (익은 딸기/안익은 딸기)
validation_dir = 'C:/Users/antl/Desktop/berry riped data/validation'  # 검증용 데이터 경로 (익은 딸기/안익은 딸기)

# 이미지 데이터 전처리 (ImageDataGenerator를 통해 데이터 증강 적용)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

# 데이터 로딩
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# CNN 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 이진 분류 (익은 딸기 vs 안익은 딸기)
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001),  # lr 대신 learning_rate 사용
              loss='binary_crossentropy',
              metrics=['accuracy'])



# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # 한 epoch당 학습할 스텝 수 (학습 데이터셋 크기에 맞게 설정)
    epochs=30,  # 에포크 수
    validation_data=validation_generator,
    validation_steps=50  # 검증 데이터셋 크기에 맞게 설정
)

# 모델 저장
model.save('strawberry_classifier.h5')

# 모델 평가
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy*100:.2f}%')
