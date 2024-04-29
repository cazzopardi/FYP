import math
import pickle

import numpy as np
import tensorflow as tf
from keras.src.engine import data_adapter
from tensorflow.python.keras import backend as tf_backend
from thundersvm import OneClassSVM

class SAE_OCSVM(tf.keras.Model):
    def __init__(self, learning_rate:float, n_input: int, num_hidden_1: int, num_hidden_2: int, num_hidden_3: int, nu=0.5, ocsvm_gpu=0) -> None:
        super().__init__()
        self.encoder: tf.keras.Sequential = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden_1, activation='tanh'),
            tf.keras.layers.Dense(num_hidden_2, activation='tanh'),
            tf.keras.layers.Dense(num_hidden_3, activation='tanh', dtype='float32')
        ])

        self.decoder: tf.keras.Sequential = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden_2, activation='tanh'),
            tf.keras.layers.Dense(num_hidden_1, activation='tanh'),
            tf.keras.layers.Dense(n_input, activation='tanh')
        ])

        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate)
        self.compile(optimizer=self.optimizer, loss='mean_squared_error')

        self.clf = OneClassSVM(nu=nu, kernel="rbf", gamma=0.1, gpu_id=ocsvm_gpu, max_mem_size=8192)

    def infer_similarity(self, X: tf.Tensor, batch_size:int=300) -> np.ndarray:
        encoded_data = self.encoder.predict(X, batch_size=batch_size)
        similarities = []
        for i in range(math.ceil(len(X)/batch_size)):
            start = i*batch_size
            end = min(i*batch_size+batch_size, len(X))
            similarities.append(self.clf.decision_function(encoded_data[start:end]))  # type: ignore
        return np.concatenate(similarities)
        # return self.clf.decision_function(self.encoder(X))
    
    # def infer_class(self, X: tf.Tensor, batch_size:int=300):
    #     encoded_data = self.encoder.predict(X, batch_size=batch_size)
    #     similarities = []
    #     for i in range(math.ceil(len(X)/batch_size)):
    #         start = i*batch_size
    #         end = min(i*batch_size+batch_size, len(X))
    #         similarities.append(self.clf.predict(encoded_data[start:end]))  # type: ignore
    #     return np.concatenate(similarities)
    
    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            z_train = self.encoder(x, training=True)
            recon_batch = self.decoder(z_train, training=True)
            re_batch = self.compute_loss(x, y, recon_batch, sample_weight)
            ae_loss = re_batch + tf.constant(10.0, dtype='float32') * tf.reduce_mean(tf.square(z_train))
        self._validate_target_and_loss(y, ae_loss)
        self.optimizer.minimize(ae_loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, recon_batch, sample_weight)

    def train(self, X_train: np.ndarray|tf.Tensor, epochs=10, batch_size=100):
        if type(X_train) == np.ndarray:
            X_train = tf.convert_to_tensor(X_train, dtype='float32')
        self.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, verbose=1)
        encoded_data = self.encoder.predict(X_train, batch_size=batch_size)
        tf_backend.clear_session()
        print('Fitting OCSVM')
        self.save_weights('dl_model')
        # self.load_weights('dl_model')
        self.clf.fit(encoded_data)
        print('Training complete.')
    
    def save(self, path:str):
        self.save_weights(path)
        self.clf.save_to_file(path+'.sk')

    def load(self, path: str):
        self.load_weights(path)
        self.clf.load_from_file(path+'.sk')
