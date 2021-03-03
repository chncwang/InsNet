N3LDG++
===========================
N3LDG++ is a neural network libary based on dynamic graph for natural language processing.

I'm the author to implement the first GPU implementation of N3LDG and to develop a GPU faster and well-designed N3LDG, we derive it independently, thereby being able to develop it according to our own ideas and remvove some original C++ codes of bad taste, without the restriction of being compatible of those N3LDG-based repos.

Due to incorrect operations of git, earlier tens of commitments by me(mainly about how CUDA codes are implemented step by step from scratch are eliminated), and these commitments can be seen in another repo https://github.com/chncwang/N3LDG

GPUs are expensive and it is important to make full use of GPU threads. Compared with N3LDG, all CUDA codes are especially designed for NLP net, to achieve an extremely fast training speed. In addition, our CUDA implementation is completed.

Now N3LDG++ is a much better N3LDG in terms of GPU support, Graph execution efficiency, easy APIs, etc. We will write wiki sometime.

## Installation:
### Prerequisitions:

boost 1.68(a fairly late version may be satisfied) is required.

CUDA 8.0 is required if you want to run it on GPU.

If you have any question, feel free to send an email to chncwang@gmail.com

## Examples:
Some examples are realeased at:
* https://github.com/chncwang/single-turn-conversation

for cuda usage, see:
* https://github.com/chncwang/news-title-classification
* https://github.com/chncwang/single-turn-conversation

## Paper:
Since N3LDG++'s paper is not completed yet, please cite

Wang, Qiansheng, Nan Yu, Meishan Zhang, Zijia Han, and Guohong Fu. "N3LDG: A Lightweight Neural Network Library for Natural Language Processing." Beijing Da Xue Xue Bao 55, no. 1 (2019): 113-119.

## Authors:
Wang Qiansheng, Zhang Meishan, Wang Zhen, Han Zijia, Cao Xue
