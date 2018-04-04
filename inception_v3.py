import robustml
from utils import optimistic_restore
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import functools
import os

class InceptionV3(robustml.model.Model):
    def __init__(self, sess):
        self._sess = sess
        self._input = tf.placeholder(tf.float32, (299, 299, 3))
        input_expanded = tf.expand_dims(self._input, axis=0)
        self._logits, self._predictions = _model(sess, input_expanded)
        self._dataset = robustml.dataset.ImageNet((299, 299, 3))
        # NOTE: the InceptionV3 paper doesn't claim any robustness under any
        # particular threat model; we are choosing one here for illustrative
        # purposes. If you're defining your own robust model, you should choose
        # a concrete threat model here, so that you're making a testable claim
        # about robustness under a well-defined model.
        self._threat_model = robustml.threat_model.Linf(epsilon=0.01)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        return self._sess.run(self._predictions, {self._input: x})[0]
    
    # exposing some internals to make it less annoying for attackers to do a
    # white-box attack

    @property
    def input(self):
        return self._input

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions

def _get_model(reuse):
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    func = nets.inception.inception_v3
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn

def _preprocess(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

_INCEPTION_CHECKPOINT_NAME = 'inception_v3.ckpt'
INCEPTION_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    _INCEPTION_CHECKPOINT_NAME
)

# input is [batch, _, _, 3], pixels in [0, 1]
# output is [batch, 1000]
_inception_initialized = False
def _model(sess, image):
    global _inception_initialized
    network_fn = _get_model(reuse=_inception_initialized)
    size = network_fn.default_image_size
    preprocessed = _preprocess(image, size, size)
    logits, _ = network_fn(preprocessed)
    logits = logits[:,1:] # ignore background class
    predictions = tf.argmax(logits, 1)

    if not _inception_initialized:
        optimistic_restore(sess, INCEPTION_CHECKPOINT_PATH)
        _inception_initialized = True

    return logits, predictions
