Entity Mention Detection using SENNA

This is Torch implementation of the SENNA system introduced in the following paper:
R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu and P. Kuksa. Natural Language Processing (Almost) from Scratch, Journal of Machine Learning Research (JMLR), 2011. 

This code is tuned for identifying chemical entity mention in patents (task specified in http://www.biocreative.org/tasks/biocreative-v/cemp-detailed-task-description/)

Deep Neural Network - Specifications
Approach: Sentence-based (using convolution)
Pooling used: Max-pooling
Loss function: Maximizing log likelihood
Optimization: Stochastic Gradient Descent (SGD)