import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import math
from collections import Counter

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created: {directory}")

def build_advanced_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train():
    base_dir = r'C:\Users\Пользователь\Desktop\pbid'
    train_dir = os.path.join(base_dir, 'data', 'train')
    model_dir = os.path.join(base_dir, 'models')
    log_dir = os.path.join(model_dir, 'logs')
    
    ensure_dir(model_dir)
    ensure_dir(log_dir)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.15
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    class_counts = Counter(train_generator.classes)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples/(len(class_counts) * count) for cls, count in class_counts.items()}
    
    model = build_advanced_model(len(train_generator.class_indices))
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=math.ceil(train_generator.samples / train_generator.batch_size),
        validation_data=val_generator,
        validation_steps=math.ceil(val_generator.samples / val_generator.batch_size),
        epochs=100,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    model.save(os.path.join(model_dir, 'final_model.h5'))
    print("✅ Training has completed!")
train()