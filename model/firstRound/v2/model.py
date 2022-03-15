from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense
from keras.optimizers import Adam

def create_model(input_shape, summary=False):
    
    # Input layer
    X_input = Input(shape=input_shape)
       
    # Dense layer
    X = BatchNormalization()(X_input)
    X = Dense(2**11, activation='relu')(X)
    
    # Dense layer
    X = BatchNormalization()(X)
    X = Dense(2**9, activation='relu')(X)
        
    # Dense layer
    X = BatchNormalization()(X)
    X = Dense(2**7, activation='relu')(X)
    
    # Dense layer
    X = BatchNormalization()(X)
    X = Dense(2**5, activation='relu')(X)
    
    # Dense layer
    X = BatchNormalization()(X)
    X = Dense(2**3, activation='relu')(X)
    
    # Dense layer
    X = BatchNormalization()(X)
    X = Dense(1, activation='relu')(X)
    
    model = Model(inputs=X_input, outputs=X)
    
    if summary:
        model.summary()
    
    return model