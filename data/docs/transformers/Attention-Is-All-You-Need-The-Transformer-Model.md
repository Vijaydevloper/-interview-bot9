# Attention Is All You Need: The Transformer Model

This presentation explores the groundbreaking Transformer architecture, a novel neural network model based entirely on attention mechanisms. The Transformer replaces traditional recurrent and convolutional layers with multi-headed self-attention, enabling faster training and superior translation quality.

We will cover the model's design, training, results, and its impact on machine translation and other NLP tasks, supported by visualizations and performance comparisons.

# Introduction to the Transformer Architecture

Key Innovation

The Transformer discards recurrence and convolutions, relying solely on attention to model global dependencies between input and output sequences.

This design allows for significantly more parallelization and faster training compared to traditional recurrent neural networks.

Encoder-Decoder Structure

Both encoder and decoder consist of stacked layers combining multi-head self-attention and fully connected feed-forward networks.

Residual connections and layer normalization enhance training stability and performance.

# Transformers Internal Structure



# Architecture



# Positional Encoding

Since the Transformer lacks recurrence, it adds sinusoidal positional encodings to input embeddings to inject sequence order information.

This approach enables the model to learn relative positions and generalize to longer sequences.

# Words to Vector Embeddings



# Attention Mechanisms Explained

Scaled Dot-Product Attention

Computes attention weights by scaling dot products of queries and keys, followed by a softmax to weight values.

Multi-Head Attention

Performs multiple attention operations in parallel on linearly projected queries, keys, and values, allowing the model to attend to different representation subspaces.

# Self - Attention Mechanism



# Attention Mechanism Formula



# Attention Mechanisms Overview



# Transformer Components: Encoder and Decoder

Encoder Stack

6 identical layers

Each with multi-head self-attention and feed-forward sub-layers

Residual connections and layer normalization applied

Decoder Stack

6 identical layers

Includes an additional multi-head attention layer over encoder outputs

Masked self-attention prevents attending to future positions

# Feed-Forward Networks

Each layer includes a position-wise fully connected feed-forward network with two linear transformations and a ReLU activation.

This enhances the model's capacity to process information independently at each position.

# Multilayer Perceptron



# Training Setup and Optimization

Datasets

Trained on WMT 2014 English-German (4.5M sentence pairs) and English-French (36M sentences) datasets using byte-pair and word-piece encoding.

Hardware & Schedule

Training used 8 NVIDIA P100 GPUs, with base models trained for 12 hours and big models for 3.5 days.

Optimizer & Regularization

Adam optimizer with learning rate warmup and decay, dropout applied to sub-layers and embeddings, and label smoothing to improve accuracy.

# Performance and Results in Machine Translation

28.4

English-German BLEU

Transformer big model achieves state-of-the-art BLEU score, surpassing previous models by over 2 points.

41.8

English-French BLEU

New single-model state-of-the-art BLEU score with significantly reduced training cost.

12h

Training Time

Base models train in as little as 12 hours on 8 GPUs.

# Model Variations and Ablation Studies

Parameter

Effect on BLEU

Number of Attention Heads

Too few or too many heads reduce quality; 8 heads optimal

Attention Key Size

Smaller key size hurts model quality

Dropout Rate

Higher dropout prevents overfitting

Positional Encoding

Sinusoidal and learned embeddings yield similar results

# Generalization to English Constituency Parsing

The Transformer was evaluated on English constituency parsing, a task with structural constraints and longer outputs.

It outperformed many previous models, including RNN-based approaches, achieving competitive F1 scores on the Penn Treebank dataset.

Model

WSJ 23 F1 Score

Transformer (4 layers, semi-supervised)

92.7

RNN Grammar (generative)

93.3

Berkeley-Parser

90.4

# Conclusion and Future Directions

Innovation

The Transformer replaces recurrence with multi-headed self-attention, enabling faster and more parallelizable sequence modeling.

Impact

Achieves state-of-the-art results in machine translation and generalizes well to other NLP tasks.

Future Work

Plans include applying attention models to modalities beyond text and exploring local attention for large inputs like images and audio.
