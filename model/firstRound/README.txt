v1 -> v2: in order to reduce bias, implementing bigger network only with dense layers.
v2 -> v3: bigger network with dense layers failed; therefore, implementing RNN model.
v3 -> v4: Adding Dropout to reduce overfitting; predicting the output of test set given in August 31.
v4 -> v5: Implimenting 2 LSTM layers in order to implement one more dropout layer.
v5 -> v6: Adding a dropout layer after input layer to v4, recording evaluation loss, and retraining to determing the early stopping.