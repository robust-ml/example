import robustml
import sys
import tensorflow as tf
import numpy as np

class InceptionV3PGDAttack(robustml.attack.Attack):
    def __init__(self, sess, model, epsilon, max_steps=100, learning_rate=0.001, debug=False):
        self._sess = sess
        self._model = model
        self._epsilon = epsilon
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._debug = debug

        self._label = tf.placeholder(tf.int32, ())
        one_hot = tf.expand_dims(tf.one_hot(self._label, 1000), axis=0)
        self._loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=one_hot)
        self._grad, = tf.gradients(self._loss, model.input)

    def run(self, x, y, target):
        mult = -1
        if target is None:
            target = y
            mult = 1
        adv = np.copy(x)
        lower = np.clip(x - self._epsilon, 0, 1)
        upper = np.clip(x + self._epsilon, 0, 1)
        for i in range(self._max_steps):
            p, l, g = self._sess.run(
                [self._model.predictions, self._loss, self._grad],
                {self._model.input: adv, self._label: target}
            )
            if self._debug:
                print(
                    'attack: step %d/%d, loss = %g (true %d, predicted %d)' % (i+1, self._max_steps, l, y, p),
                    file=sys.stderr
                )
            if p != y:
                # we're done
                if self._debug:
                    print('returning early', file=sys.stderr)
                break
            adv += mult * self._learning_rate * np.sign(g)
            adv = np.clip(adv, lower, upper)
        return adv

class InceptionV3FGSMAttack(robustml.attack.Attack):
    def __init__(self, sess, model, epsilon):
        self._sess = sess
        self._model = model
        self._epsilon = epsilon

        self._label = tf.placeholder(tf.int32, ())
        one_hot = tf.expand_dims(tf.one_hot(self._label, 1000), axis=0)
        self._loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=one_hot)
        self._grad, = tf.gradients(self._loss, model.input)

    def run(self, x, y, target):
        mult = -1
        if target is None:
            target = y
            mult = 1
        g = self._sess.run(self._grad, {self._model.input: x, self._label: y})
        adv = np.clip(x + mult * self._epsilon * np.sign(g), 0, 1)
        return adv

class NullAttack(robustml.attack.Attack):
    def run(self, x, y, target):
        return x
