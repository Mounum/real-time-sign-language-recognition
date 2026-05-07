import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ===== DATASET PATH (FIXED) =====
data_dir = "dataset"

# ===== SETTINGS =====
img_size = (224, 224)
batch_size = 32

# ===== DATA GENERATOR =====
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ===== PRINT CLASSES =====
print("Class indices:", train_data.class_indices)

# ===== MODEL (TRANSFER LEARNING) =====
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(
    train_data.num_classes,
    activation='softmax'
)(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model (faster training)
for layer in base_model.layers:
    layer.trainable = False

# ===== COMPILE =====
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===== TRAIN =====
model.fit(
    train_data,
    validation_data=val_data,
    epochs=8   # you can increase to 10 later
)

# ===== SAVE MODEL (FIXED NAME) =====
model.save("sign_language_model.h5")

print("✅ Model trained and saved as sign_language_model.h5")