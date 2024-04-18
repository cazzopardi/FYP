import numpy as np
import tensorflow as tf
from thundersvm import OneClassSVM

class SAE_OCSVM:
    def __init__(self, learning_rate:float, n_input: int, num_hidden_1: int, num_hidden_2: int, num_hidden_3: int, nu=0.5) -> None:
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden_1, activation='tanh'),
            tf.keras.layers.Dense(num_hidden_2, activation='tanh'),
            tf.keras.layers.Dense(num_hidden_3, activation='tanh')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden_2, activation='tanh'),
            tf.keras.layers.Dense(num_hidden_1, activation='tanh'),
            tf.keras.layers.Dense(n_input, activation='tanh')
        ])

        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate)

        self.clf = OneClassSVM(nu=nu, kernel="rbf", gamma=0.1)

    def train(self, X_train, num_steps=500, batch_size=100, display_step=10, display_callback=None):
        num_batch = int(X_train.shape[0]/batch_size)
        for i in range(num_steps):
            re = 0
            for i_batch in range(num_batch):
                batch_x = X_train[i_batch*batch_size:(i_batch+1)*batch_size]
                with tf.GradientTape() as tape:
                    z_train = self.encoder(batch_x, training=True)
                    recon_batch = self.decoder(z_train, training=True)
                    re_batch = tf.reduce_mean(tf.square(recon_batch - batch_x))
                    ae_loss = re_batch + 10 * tf.reduce_mean(tf.square(z_train))  # type: ignore
                    gradients = tape.gradient(ae_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables + self.decoder.trainable_variables))
                re += re_batch.numpy()  # type: ignore
    
            if i % display_step == 0 or i == 1:
                # z_train = self.encoder(X_train, training=False)
                # z_test = self.encoder(x_test, training=False)
                # fpr_ae, tpr_ae, auc_ae = AUC_AE(x_test, y_test)
                # fpr_svm, tpr_svm, auc_svm = AUC_SVM(z_train, z_test, y_test)
                # print(f'Step {i}: Minibatch Loss: {re/num_batch:.4f} - AUC_AE {auc_ae:.3f} - AUC_SVM:{auc_svm:.3f}')
                print(f'Step {i}: Minibatch Loss: {re/num_batch:.4f}')
                if display_callback:
                    display_callback(self)
        self.clf.fit(self.encoder(X_train))
    def predict(self, X):
        return self.clf.decision_function(self.encoder(X))