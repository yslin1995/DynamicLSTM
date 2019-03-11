# Long Short-Term Memory with Dynamic Skip Connections

TensorFlow implementation of [Long Short-Term Memory with Dynamic Skip Connections](https://arxiv.org/pdf/1811.03873.pdf) (AAAI 2019).

![Model](.\img\model.png)

The code is partially referred to https://github.com/tensorflow/models.

## Requirements

- Python 2.7 or higher
- Numpy
- TensorFlow 1.0



## Usage

### Language Modeling 

```
$ python RL_train.py --rnn_size 650 --rnn_layers 2 --lamda 1 --n_actions 5
```



### Number Prediction

First, you should run the following command to produce train/validation/test data for the task of number prediction. (Please refer to the paper for a detailed description) 

```
$ python produce_data.py
```

Later, you can reproduce the results in the paper by executing:

```
$ python train_RL10_1.py
```



## Citation

If you find this code useful, please cite us in your work:

```latex
@article{gui2018long,
  title={Long Short-Term Memory with Dynamic Skip Connections},
  author={Gui, Tao and Zhang, Qi and Zhao, Lujun and Lin, Yaosong and Peng, Minlong and Gong, Jingjing and Huang, Xuanjing},
  journal={arXiv preprint arXiv:1811.03873},
  year={2018}
}
```

