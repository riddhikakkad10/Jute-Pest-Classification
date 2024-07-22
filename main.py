import tensorflow as tf
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint # type: ignore

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set image dimensions
img_width, img_height = 224, 224  # Adjust to match ResNet50 input dimensions

# Paths to the dataset
train_dir = '/Users/riddhikakkad/Desktop/Jute_Pest_Dataset/train'
val_dir = '/Users/riddhikakkad/Desktop/Jute_Pest_Dataset/val'
test_dir = '/Users/riddhikakkad/Desktop/Jute_Pest_Dataset/test'

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=64,  # Increase batch size if memory allows
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Load the ResNet50 model with locally downloaded weights
local_weights_file = '/Users/riddhikakkad/Desktop/Jute_Pest_Dataset/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = ResNet50(weights=local_weights_file, include_top=False, input_shape=(img_width, img_height, 3))

# Add custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(17, activation='softmax', dtype='float32')(x)  # Set dtype to float32

# Create the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze some of the top layers of the base model
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Use a learning rate scheduler to reduce learning rate when the model plateaus
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# Save the best model during training
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,  # Increase the number of epochs
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[lr_scheduler, checkpoint]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_accuracy:.4f}')