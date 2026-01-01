import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from .config import INPUT_SHAPE, NUM_CLASSES

from tensorflow.keras.regularizers import l2

def build_model():
    """
    Constructs the Emotion Recognition model based on MobileNetV2.
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE
    )

    # Freeze base model layers initially (optional, but good for transfer learning)
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
