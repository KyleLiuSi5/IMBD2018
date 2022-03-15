from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, Dense, Flatten
from keras.optimizers import Adam

def create_model(input_shape, summary=False):
    
    # Input layer
    X_input = Input(shape=input_shape)
    
    # Conv2D layer
    X = BatchNormalization()(X_input)
    X = Conv2D(filters=1, kernel_size=4)(X)
    
    # Dense layer
    X = Flatten()(X)
    X = BatchNormalization()(X)
    X = Dense(2048, activation='relu')(X)
    
    # Dense layer
    X = BatchNormalization()(X)
    X = Dense(512, activation='relu')(X)
    
    # Dense layer
    X = BatchNormalization()(X)
    X = Dense(128, activation='relu')(X)
    
    # Dense layer
    X = BatchNormalization()(X)
    X = Dense(1, activation='relu')(X)
    
    model = Model(inputs=X_input, outputs=X)
    
    if summary:
        model.summary()
    
    return model