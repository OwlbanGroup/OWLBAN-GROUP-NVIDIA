"""
Test script to check GPU availability for TensorFlow
"""
import tensorflow as tf
import sys

def test_gpu():
    """Test if GPU is available and can be used"""
    print("TensorFlow version:", tf.__version__)

    # Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU devices found: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
        try:
            # Test GPU usage
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0]])
                b = tf.constant([[4.0], [5.0], [6.0]])
                c = tf.matmul(a, b)
                print("GPU test successful:", c.numpy())
            return True
        except RuntimeError as e:
            print(f"GPU test failed: {e}")
            return False
    else:
        print("No GPU devices found. Using CPU.")
        return False

if __name__ == "__main__":
    gpu_available = test_gpu()
    sys.exit(0 if gpu_available else 1)
