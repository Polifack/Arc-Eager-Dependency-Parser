# Arc Eager Dependency Parser
Implementation of transition-based arc-eager parser following the guidelines of [Algorithms for Deterministic Incremental Dependency Parsing](https://aclanthology.org/J08-4003.pdf). Developed with Python and Tensorflow-Keras libraries. Includes custom Conll-u libraries to read, write and modify dependency trees. As the arc-eager parser cannot deal with non projective trees (trees with crossing arcs), whenever reading for training a treebank those kind of trees will be removed. Note that the multi-expression nodes (e.g. 2-3) and empty nodes (. or _) are not supported.

# Requirements

A custom script for create conda enviroment is provided in install.sh. It will create a conda enviroment with the following libraries:

- Python 3.8
- Tensorflow 2.9.2
- Keras 2.9.0
- Numpy 
- Matplotlib

# Usage

## Training

To train the model, run the following command:

```bash
python main.py train
    --input <input_folder> 
    --output <output_folder> 
    --seq_l_s <ns> 
    --seq_l_b <nb>
    --epochs <epochs> 
    --batch_size <batch_size> 
    --lr <lr> 
    --hidden_size <hidden_size> 
    --embedding_size <embedding_size> 
```

Where:

- input_folder: Folder containing the training data in conll-u format.
- output_folder: Folder where the model will be saved.
- ns: Sequence length from the stack. Number of words from stack to be processed at the same time.
- nb: Sequence length from the buffer. Number of words from buffer to be processed at the same time.
- epochs: Number of epochs to train the model.
- batch_size: Batch size.
- lr: Learning rate.
- hidden_size: Hidden size of the Dense layer.
- embedding_size: Words and tags embedding size.

## Evaluation

To evaluate the model, run the following command:

```bash
python main.py eval
    --input <input_folder> 
    --model <model_folder> 
    --output <out_folder> 
    [--postprocess default=True]
    [--multi-root  default=False]
```

Where:

- input_folder: Folder containing the test data in conll-u format.
- model_folder: Folder containing the model to be evaluated.
- out_folder: Folder where the output will be saved.
- postprocess: If present, the output will be postprocessed (remove cycles, out of bound heads).

# Implementation

## Model

The model is a simple feed-forward neural network with the following layers:

- Embedding layer: Embedding layer for words and tags.
- Dense layer: Dense layer with hidden size neurons.

The model is trained by default with the Adam optimizer and the categorical crossentropy loss function. The model architecture is the following:
![Model architecture](http://github.com/polifack/Arc-Eager-Dependency-Parser/blob/main/pics/model_architecture.png)

## Transitions

The following transitions are implemented:

- SHIFT: Add the first word from the buffer to the stack.
- LEFT-ARC: Add a dependency from the first word in the stack to the second word in the stack.
- RIGHT-ARC: Add a dependency from the second word in the stack to the first word in the stack.
- REDUCE: Remove the first word from the stack.

## Code structure

The code is structured as follows:

- main.py: Main script. Contains the training and evaluation functions.
- ArcEagerModel.py: Contains the machine-learning model definition and deals with creation of tokenizer dictionaries.
- ArcEager.py: Contains the parser transition logic.
- ArcEagerConfig.py: Contains the parser configuration logic (Stack, Buffer, etc).
- ConllTree.py, ConllNode.py : Contains the conll-u libraries.

# Results

Results are obtained using the evaluation script provided by the CoNLL 2018 shared task. We will employ the LAS and UAS metrics. We will perform experiments for different sequence lengths and different hidden sizes on the English-ParTUT dataset. We employed a categorical cross-entropy loss function, a learning rate of 0.001 with adam optimizer and a 128-sized embeddings layer for both words and part of speech tags. The activation functions for all the experiments are relu and softmax for the hidden dense layer and the output dense layers, respectively. In order to measure the speed of the parser we employed Google Colab free plan machine. The first batch of experiments employed the same number of words taken from the stack and buffer and the results can be seen in table (a). The second batch of experiments were performed employing only a 256 dense dimension
and tweaked the number of words taken from the stack and buffer separately and can be seen in (b).

![Results](http://github.com/polifac/Arc-Eager-Dependency-Parser/blob/main/pics/results.png)

We also included the Google Colab notebook used for the experiments in the repository. They can be found in the notebooks folder.

# References

- [Algorithms for Deterministic Incremental Dependency Parsing](https://aclanthology.org/J08-4003.pdf)
- [CoNLL 2018 Shared Task](https://universaldependencies.org/conll18/)
- [A fast and accurate dependency parser using neural networks](https://aclanthology.org/D14-1082/)



