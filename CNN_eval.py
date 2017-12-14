import tensorflow as tf
import numpy as np
from CNN import cnn_model_fn

def main(unused_argv):

    landmarks= np.load('./test/test_landmarks.npy')
    fft =  np.load('./test/test_fft.npy')
    eval_data = fft  
    eval_labels = landmarks

    speech_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./record")
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":eval_data},shuffle=False)
    eval_results = speech_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()
