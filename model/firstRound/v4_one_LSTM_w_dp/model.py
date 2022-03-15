from keras.models import Model
from keras.layers import Input, BatchNormalization, Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam

def create_model(input_shape, summary=False):
    
    # Input layer
    X_input = Input(shape=input_shape)
        
    # Bidirectional LSTM layer
    X = BatchNormalization()(X_input)
    X = Bidirectional(LSTM(128, return_sequences=False))(X)
        
    # Dense layer
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    X = Dense(1, activation='relu')(X)
        
    model = Model(inputs=X_input, outputs=X)
    
    if summary:
        model.summary()
    
    return model