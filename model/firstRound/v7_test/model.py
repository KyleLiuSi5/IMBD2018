from keras.models import Model
from keras.layers import Input, BatchNormalization, Bidirectional, LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam

def create_model(input_shape, summary=False):
    
    # Input layer
    X_input = Input(shape=input_shape)
    
    # Conv2D layer - 64
    X = BatchNormalization()(X_input)
    X = Conv2D(filters=64, kernel_size=(4, 4), activation='relu')(X)
    #X = Conv2D(filters=32, kernel_size=(4, 1), activation='relu')(X)
    X = MaxPooling2D(pool_size=(4, 1), strides=4)(X)
    
    # Conv2D layer - 128
    X = BatchNormalization()(X)
    X = Conv2D(filters=128, kernel_size=(4, 1), activation='relu')(X)
    #X = Conv2D(filters=64, kernel_size=(4, 1), activation='relu')(X)
    X = MaxPooling2D(pool_size=(4, 1), strides=4)(X)
    
    # Conv2D layer - 256
    #X = BatchNormalization()(X)
    #X = Conv2D(filters=128, kernel_size=(4, 1), activation='relu')(X)
    #X = Conv2D(filters=128, kernel_size=(4, 1), activation='relu')(X)
    #X = Conv2D(filters=128, kernel_size=(4, 1), activation='relu')(X)
    #X = MaxPooling2D(pool_size=(4, 1), strides=4)(X)
    
    # Bidirectional LSTM layer
    X = Reshape((467, 128))(X)
    X = BatchNormalization()(X)
    X = Bidirectional(LSTM(64, return_sequences=False))(X)
    
    # Dense layer
    X = BatchNormalization()(X)
    X = Dense(1, activation='relu')(X)
        
    model = Model(inputs=X_input, outputs=X)
    
    if summary:
        model.summary()
    
    return model