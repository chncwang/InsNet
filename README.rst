.. InsNet documentation master file, created by
   sphinx-quickstart on Sun Mar 21 22:16:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

InsNet documentation
===================================

`InsNet <https://github.com/chncwang/insnet>`_ (`documentation <https://insnet.readthedocs.io/en/latest>`_) is a powerful neural network library aiming at building instance-dependent computation graphs. It is designed to support padding-free dynamic batching, thus allow users to focus on building the model for a single instance. This design has at least four advantages as follows:

1. **It can batch not only operators in a mini-batch but also operators in the same instance.** For example, it can batch two parallel transformers from the same instance.
2. **It makes it super easy to build NLP models with instance-dependent computation graphs and execute them in batch**, such as tree-LSTM  and `hierarchical Transformers <https://www.aclweb.org/anthology/P19-1500.pdf>`_.
3. **It reduces users' intellectual burden of manual batching**, as InsNet can efficiently take over all batching procedures. As such, users even need not know the concept of tensor, but only the matrix and vector (which is a one-column matrix), neither the concept of padding.
4. **It significantly reduces memory usage since no padding is needed and lazy execution can release useless tensors immediately.**

To summarize, we believe that **Padding-free Dynamic Batching is the feature that NLPers will dive into but is surprisingly not supported by today's deep learning libraries**.

Besides, InsNet has the following features:

1. It is written in C++ 14 and is built as a static library.
2. For GPU computation, we write almost all CUDA kernels by hand, allowing efficient parallel computation for matrices of unaligned shapes.
3. Both lazy and eager execution is supported, with the former allowing for automatic batching and the latter facilitating users' debugging.
4. For the moment, it provides about thirty operators with both GPU and CPU implementations, supporting building modern NLP models for sentence classification, sequence tagging, and language generation. It furthermore provides NLP modules such as attention, RNNs, and the Transformer, built with the aforementioned operators.

Studies using InsNet are listed as follows, and we are looking forward to enriching this list:

- `Unseen Target Stance Detection with Adversarial Domain Generalization <https://arxiv.org/pdf/2010.05471.pdf>`_
- `Cue-word Driven Neural Response Generation with a Shrinking Vocabulary <https://arxiv.org/pdf/2010.04927.pdf>`_

InsNet uses *Apache 2.0* license allowing you to use it in any project. But if you use InsNet for research, please cite this paper as follows and declare it as an early version of InsNet since the paper of InsNet is not completed yet::

  @article{wang2019n3ldg,
  title={N3LDG: A Lightweight Neural Network Library for Natural Language Processing},
  author={Wang, Qiansheng and Yu, Nan and Zhang, Meishan and Han, Zijia and Fu, Guohong},
  journal={Beijing Da Xue Xue Bao},
  volume={55},
  number={1},
  pages={113--119},
  year={2019},
  publisher={Acta Scientiarum Naturalium Universitatis Pekinenis}
  }

Due to incorrect Git operations, the very early history of InsNet is erased, but you can see it in another `repo <https://github.com/chncwang/N3LDG>`_.

If you have any question about InsNet, feel free to post an issue or send me an email: chncwang@gmail.com

See the `documentation <https://insnet.readthedocs.io/en/latest>`_ for more details.
