## Entity Mention Detection using SENNA<br />

This is Torch implementation of the SENNA system introduced in the following paper:
R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu and P. Kuksa. Natural Language Processing (Almost) from Scratch, Journal of Machine Learning Research (JMLR), 2011. <br />

This code is tuned for identifying chemical entity mention in patents (task specified in http://www.biocreative.org/tasks/biocreative-v/cemp-detailed-task-description/)
<br />

*To execute it,* <br />
**cd code/** <br />
**th main.lua** <br />
<br />
*To know configurable parameters of the model,* <br />
**th main.lua -help** <br />
<br />
*Prerequisites to run:* <br />
1. Torch 7 <br />
2. nn <br />
3. optim <br />
4. xlua <br />
5. cunn (if you are running in a gpu) <br />
Packages [2-5] can be installed using Luarocks. <br />
For example,<br />
*To install nn,* <br />
luarocks install nn <br />
