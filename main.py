import tensorflow as tf   # TensorFlow registers PluggableDevices here.
tf.config.list_physical_devices('GPU')  # APU device is visible to TensorFlow.
