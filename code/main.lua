--[[

SENNA system for Chemical Mention Detection

]]--

require 'torch'
require 'io'
require 'nn'
require 'sys'
require 'optim'
require 'os'
require 'xlua'
require 'lfs'
include('senna.lua')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Identify chemical named entity mentions using SENNA system')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-train_file','train.tsv','training set file location')
cmd:option('-dev_file','dev.tsv','dev set file location')
cmd:option('-test_file','test.tsv','test set file location')
cmd:option('-to_lower',1,'change the case of word to lower case')
-- model params (general)
cmd:option('-wdim',300,'dimensionality of word embeddings')
cmd:option('-min_freq',5,'words that occur less than <int> times will not be taken for training')
cmd:option('-pre_train',1,'initialize word embeddings with pre-trained vectors?')
cmd:option('-wwin',5,'word convolution units')
cmd:option('-hid_size',300,'hidden units')
-- optimization
cmd:option('-learning_rate',0.1,'learning rate')
cmd:option('-max_epochs',1,'number of full passes through the training data')
cmd:option('-reg',1e-4,'regularization parameter l2-norm')
cmd:option('-pre_train',0,'initalize word embeddings with pre-trained word vectors')
cmd:option('-clip',5,'threshold to clip gradients')
-- GPU/CPU
cmd:option('-gpu',1,'1=use gpu; 0=use cpu;')
-- Book-keeping
cmd:option('-print_params',0,'output the parameters in the console. 0=dont print; 1=print;')

-- parse input params
params=cmd:parse(arg)

if params.print_params==1 then
	-- output the parameters	
	for param, value in pairs(params) do
	    print(param ..' : '.. tostring(value))
	end
end
if params.gpu==1 then
	require 'cutorch'
	require 'cunn'
	require 'cunnx'
end

model=Senna(params)
model:train()