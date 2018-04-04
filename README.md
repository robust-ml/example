# Example implementation of robustml interface

This repository contains an example implementation of a [robustml][repo] model
and attack interface. It also demonstrates how to evaluate a particular attack
against a particular defense.

See `inception_v3.py` to see how to implement the `Model` interface. See
`attack.py` to see how to implement the attack interface. See `run.py` to see
how to run a particular attack against a particular defense.

## Dependencies

This code depends on robustml. You can install it from PyPI with `pip install
robustml` or you can clone the [repo] and install it with `pip install -e .`.

## Running

Run with:

```
python run.py --imagenet-path <path to imagenet data>
```

Run `python run.py --help` to see more usage details.

[repo]: https://github.com/robust-ml/robustml
