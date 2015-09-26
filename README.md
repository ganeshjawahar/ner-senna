Entity Mention Detection using SENNA

This is Torch implementation of the SENNA system introduced in the following paper:
R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu and P. Kuksa. Natural Language Processing (Almost) from Scratch, Journal of Machine Learning Research (JMLR), 2011. 

This code is tuned for identifying chemical entity mention in patents (task specified in http://www.biocreative.org/tasks/biocreative-v/cemp-detailed-task-description/)

To execute it,

cd code/
th main.lua

To get help of configuruable parameters of the model,
th main.lua -help

Prerequisites to run:
1. Torch 7
2. nn
3. optim
4. xlua
5. cunn (if you are running in a gpu)
Packages [2-6] can be installed using Luarocks.
For examples, to install nn,
luarocks install nn
