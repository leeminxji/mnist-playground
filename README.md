# MNIST-PLAYGROUND

## Prerequisite

- pyenv (python version 3.7.11)

```bash
pyenv install 3.7.11

cd path/to/repo
pyenv local 3.7.11
# or
pyenv virtualenv 3.7.11 mnist-playground
pyenv local mnist-playground

pyenv which python # ~/.pyenv/shims/python or ~/.pyenv/versions/mnist-playground/bin/python
python -V # Python 3.7.11

pip install -r requirements.txt
```

## Development (...ing)

```bash
# create trained model file
cd path/to/repo
./scripts/run_model.sh

# predict image
./scripts/run_main.sh
```

## Reference

https://www.tensorflow.org/tutorials/keras/classification?hl=ko
