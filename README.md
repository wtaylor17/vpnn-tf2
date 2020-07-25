# vpnn-tf2
Tensorflow 2.x implementation of Volume-Preserving Feed-Forward Neural Networks

# Why VPNNs?
* All transformations preserve volume, helping maintain gradient
magnitudes without the stronger restriction of unitary-ness
* All sublayers' forward passes are computed in `O(n)` time,
very fast for production environments
* Comparable performance to standard FF layers, but you can have way
more of them without hurting training or runtime nearly as much

# A relevant `pip freeze` from the dev computer:
```bash
Keras-Preprocessing==1.1.2
numpy==1.19.0
oauthlib==3.1.0
opt-einsum==3.2.1
scipy==1.4.1
six==1.15.0
tensorboard==2.2.2
tensorboard-plugin-wit==1.7.0
tensorflow==2.2.0
tensorflow-estimator==2.2.0
```

# Usage

```python
from vpnn import vpnn

vpnn_model = vpnn(...) # build the model
# train, do whatever...
```

# Get started
* Run `pip install .` from the repo home directory
* Change directories to `./demos`
* Run the following:
```
python mnist.py --layers 2 --rotations 8 --epochs 30 --batch_size 64 --tensorboard --save_checkpoints
```
* Mess around. The models are saved in the TF `SavedModel` format automatically.

# TODOS
- [x] implement rotational sublayer
- [x] implement permutation sublayer
- [x] implement diagonal sublayer
- [x] implement bias sublayer
- [x] implement chebyshev activation
- [x] implement trainable chebyshev
- [x] make MNIST demo
- [x] compare with paper results on MNIST ([see the notebook here](https://colab.research.google.com/drive/1XunWlmccY4IIMwMtCHTPJWG6idgRKGvQ?usp=sharing))
- [ ] make IMDB demo
- [ ] compare with paper results on IMDB

# Citation
```bibtex
@misc{macdonald2019volumepreserving,
    title={Volume-preserving Neural Networks: A Solution to the Vanishing Gradient Problem},
    author={Gordon MacDonald and Andrew Godbout and Bryn Gillcash and Stephanie Cairns},
    year={2019},
    eprint={1911.09576},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```