import robustml
import tensorflow as tf
import argparse
from inception_v3 import InceptionV3
from attack import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-path', type=str, required=True,
            help='directory containing `val.txt` and `val/` folder')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--attack', type=str, default='pgd', help='pgd | fgsm | none')
    args = parser.parse_args()

    # set up TensorFlow session
    sess = tf.Session()

    # initialize a model
    model = InceptionV3(sess)

    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    if args.attack == 'fgsm':
        attack = InceptionV3FGSMAttack(sess, model, model.threat_model.epsilon)
    elif args.attack == 'pgd':
        attack = InceptionV3PGDAttack(sess, model, model.threat_model.epsilon)
    elif args.attack == 'none':
        attack = NullAttack()
    else:
        raise ValueError('unknown attack: %s' % args.attack)

    # initialize a data provider for ImageNet images
    provider = robustml.provider.ImageNet(args.imagenet_path, (299, 299, 3))

    success_rate = robustml.evaluate.evaluate(
        model,
        attack,
        provider,
        start=args.start,
        end=args.end,
        deterministic=True,
        debug=True
    )

    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))


if __name__ == '__main__':
    main()
