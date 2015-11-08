--[[

Class for Senna containing training model and logic. 

--]]

local Senna=torch.class("Senna")
local utils=require 'utils'

-- Lua Constructor
function Senna:__init(config)
	self.train_file=config.train_file
	self.dev_file=config.dev_file
	self.test_file=config.test_file
	self.res_file=config.res_file
	self.to_lower=config.to_lower
	self.wdim=config.wdim
	self.min_freq=config.min_freq
	self.wwin=config.wwin
	self.hid_size=config.hid_size
	self.pre_train=config.pre_train
	self.learning_rate=config.learning_rate
	self.max_epochs=config.max_epochs
	self.reg=config.reg
	self.gpu=config.gpu
	self.pre_train=config.pre_train
	self.clip=config.clip

	self.vocab={} -- word frequency map
	self.index2word={}
	self.word2index={}

	-- build the vocabulary
	utils.buildVocab(self)

	-- load data to main memory
	utils.loadDataTensors(self) 

	-- build the net.
    self:build_model()

    if self.pre_train==1 then
		utils.initWordWeights(self,'/home/ganesh/Downloads/t2v/vectors.840B.300d.txt')
    end

    if self.gpu==1 then
    	self:cuda()
    end
end

function Senna:train()
	print('Training...')
	local start=sys.clock()
	local cur_batch_row=0
	local iteration=0
	local optim_state={learningRate=self.learning_rate}
	params,grad_params=self.model:getParameters()

	local idx=torch.randperm(#self.train_word_tensors)
	self.best_dev_model=self.model
	self.best_dev_score=-1.0
	for epoch=1,self.max_epochs do
		print('Epoch '..epoch..' ...')
		local epoch_start=sys.clock()
		local epoch_loss=0
		local iteration=0
		for i=1,#self.train_word_tensors do
			if i%20==0 then
				xlua.progress(i,#self.train_word_tensors)
			end
			local id=idx[i]
			for k=1,#self.train_pos_lab_tensors[id] do
				local input={self.train_word_tensors[id],self.train_pos_lab_tensors[id][k][1]}
				local label=self.train_pos_lab_tensors[id][k][2]
				-- estimate f
				local output=self.model:forward(input)
				local err=self.criterion:forward(output,label)
				epoch_loss=epoch_loss+err
				-- estimate df/dW
				local bk=self.criterion:backward(output,label)
				self.model:backward(input,bk) -- Backprop
				self.model:updateParameters(self.learning_rate)				
				if grad_params:norm()>params.clip then
					grad_params:mul(params.clip/grad_params:norm())
				end
				iteration=iteration+1
			end
		end
		xlua.progress(#self.train_word_tensors,#self.train_word_tensors)
		-- Compute dev. score
		print('Computing dev score ...')
		local tp,tn,fp,fn=0,0,0,0
		for i=1,#self.dev_word_tensors do
			xlua.progress(i,#self.dev_word_tensors)
			for k=1,#self.dev_pos_lab_tensors[i] do
				local input_tensor={self.dev_word_tensors[i],self.dev_pos_lab_tensors[i][k][1]}
				local target_tensor=self.dev_pos_lab_tensors[i][k][2]
				local output=self.model:forward(input_tensor)
				local pred=1
				if output[1]<output[2] then
					pred=2
				end
				if pred==1 and target_tensor[1]==1 then
					tn=tn+1
				elseif pred==1 and target_tensor[1]==2 then
					fn=fn+1
				elseif pred==2 and target_tensor[1]==1 then
					fp=fp+1
				else
					tp=tp+1
				end	
			end
		end
		xlua.progress(#self.dev_word_tensors,#self.dev_word_tensors)
		print(string.format('%d %d %d %d',tp,fp,tn,fn))
		local precision,recall=(tp/(tp+fp)),(tp/(tp+fn))
		local fscore=((2*precision*recall)/(precision+recall))
		print(string.format("Epoch %d done in %.2f minutes. loss=%f Dev Score=(P=%.2f R=%.2f F=%.2f)\n",epoch,((sys.clock()-epoch_start)/60),(epoch_loss/iteration),precision,recall,fscore))
		if fscore>self.best_dev_score then
			self.best_dev_score=fscore
			self.best_dev_model=self.model:clone()
		end
	end

	-- Do the final testing
	print('Computing test score ...')
	local tp,tn,fp,fn=0,0,0,0
	local start=sys.clock()
	for i=1,#self.test_word_tensors do
		xlua.progress(i,#self.test_word_tensors)
		for k=1,#self.test_pos_lab_tensors[i] do
			local input_tensor={self.test_word_tensors[i],self.test_pos_lab_tensors[i][k][1]}
			local target_tensor=self.test_pos_lab_tensors[i][k][2]
			local output=self.best_dev_model:forward(input_tensor)
			local pred=1
			if output[1]<output[2] then
				pred=2
			end
			if pred==1 and target_tensor[1]==1 then
				tn=tn+1
			elseif pred==1 and target_tensor[1]==2 then
				fn=fn+1
			elseif pred==2 and target_tensor[1]==1 then
				fp=fp+1
			else
				tp=tp+1
			end	
		end
	end
	xlua.progress(#self.test_word_tensors,#self.test_word_tensors)
	print(string.format('%d %d %d %d',tp,fp,tn,fn))
	local precision,recall=(tp/(tp+fp)),(tp/(tp+fn))
	local fscore=((2*precision*recall)/(precision+recall))
	print(string.format('Test Score=(P=%.2f R=%.2f F=%.2f)',precision,recall,fscore))
	print(string.format("Testing Done in %.2f minutes.",((sys.clock()-start)/60)))
	print(string.format("Done in %.2f seconds.",sys.clock()-start))
end

function Senna:build_model(config)
	self.model=nn.Sequential()
	self.model:add(nn.ParallelTable())
	self.model.modules[1]:add(nn.LookupTable(self.vocab_size,self.wdim))
	self.model.modules[1]:add(nn.Identity())	
	self.model:add(nn.JoinTable(2))
	self.model:add(nn.TemporalConvolution(self.wdim+1,self.hid_size,self.wwin,1))
	self.model:add(nn.Max())
	self.model:add(nn.View(-1))
	self.model:add(nn.HardTanh())
	self.model:add(nn.Linear(self.hid_size,2))
	self.model:add(nn.LogSoftMax())
	self.criterion=nn.ClassNLLCriterion(torch.Tensor{0.03,0.97})
	self.soft=nn.SoftMax()
end

function Senna:cuda()
	self.model=self.model:cuda()
	self.criterion=self.criterion:cuda()
	self.soft=self.soft:cuda()
end
