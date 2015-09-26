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
	self.grad_clip=config.grad_clip
	self.batch_size=config.batch_size
	self.max_epochs=config.max_epochs
	self.reg=config.reg
	self.gpu=config.gpu

	self.vocab={} -- word frequency map
	self.index2word={}
	self.word2index={}

	utils.buildVocab(self)

	-- build the net.
    self:build_model()

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

	feval=function(x)
		-- Get new params
		params:copy(x)

		-- Reset gradients
		grad_params:zero()

		-- loss is average of all criterions
		local loss=0

		-- Forward/Backward pass
		for bi=1,#self.batch do
			local input=self.batch[bi]
			local label=self.label_tensors[bi]

			-- estimate f
			local output=self.model:forward(input)
			local err=self.criterion:forward(output,label)
			loss=loss+err

			-- estimate df/dW
			local bk=self.criterion:backward(output,label)
			self.model:backward(input,bk) -- Backprop
		end

		loss=loss/#self.batch
		grad_params:div(#self.batch)

		-- clip gradient element-wise
		--grad_params:clamp(-self.grad_clip,self.grad_clip)

		return loss,grad_params
	end	

	self.batch={}
	utils.loadDevCorpus(self)
	for epoch=1,self.max_epochs do
		print('Epoch '..epoch..' ...')
		local epoch_start=sys.clock()
		local cur_line=0
		local epoch_loss=0
		for i,line in ipairs(self.corpus_text) do
			local tensors=utils.getFullInputTensors(self,line)
			xlua.progress(i,self.corpus_size)
			for _,tensor in ipairs(tensors) do
				table.insert(self.batch,tensor[1])
				self.label_tensors[#self.batch]=tensor[2]

				if #self.batch==self.batch_size then
					-- Train this batch
					local batch_start=sys.clock()
					iteration=iteration+1

					-- Call the optimizer
					local _,loss=optim.sgd(feval,params,optim_state)
					local train_loss = loss[1] -- the loss is inside a list, pop it
					epoch_loss=epoch_loss+train_loss

					if iteration%10==0 then collectgarbage() end

					-- clean the current batch
					for k in pairs(self.batch) do
						self.batch[k]=nil
					end
				end
			end
		end
		if #self.batch~=0 then
			optim.sgd(feval,params,optim_state)
			epoch_loss=epoch_loss+train_loss
			iteration=iteration+1
		end
		xlua.progress(self.corpus_size,self.corpus_size)
		self:compute_dev_result()
		print(string.format("Epoch %d done in %.2f minutes. loss=%f\n\n",epoch,((sys.clock()-epoch_start)/60),(epoch_loss/iteration)))
	end
	print(string.format("Done in %.2f seconds.",sys.clock()-start))
end

function Senna:compute_dev_result()
	print('Computing dev score...')
	local start=sys.clock()
	local predictions={}
	local groundtruth={}
	for i,line in ipairs(self.dev_text) do
		local tensors=utils.getFullInputTensors(self,line)
		xlua.progress(i,#self.dev_text)
		for j,tensor in ipairs(tensors) do
			local pred=self.model:forward(tensor[1])
			table.insert(predictions,utils.argmax(pred))
			table.insert(groundtruth,tensor[2][1])
		end
	end
	xlua.progress(#self.dev_text,#self.dev_text)
	local pred_tensor=torch.Tensor(#predictions)
	local gt_tensor=torch.Tensor(#groundtruth)
	print(string.format("Dev Score = %.4f",utils.accuracy(pred_tensor,gt_tensor)))
	print(string.format("Done in %.2f seconds.",sys.clock()-start))
end

function Senna:compute_test_result()
	print('Computing test result...')
	local start=sys.clock()
	local file=io.open(self.res_file,'w')
	utils.loadTestCorpus(self)
	for i,line in ipairs(self.test_text) do
		local tensors=utils.getFullInputTensors(self,line)
		xlua.progress(i,#self.test_text)
		file:write(line..'\n')
		res=''
		for j,tensor in ipairs(tensors) do
			local pred=self.model:forward(tensor[1])
			res=res..utils.argmax(pred)..' '
		end
		res=utils.trim(res)
		file:write(res..'\n')
	end
	file:close()
	xlua.progress(#self.test_text,#self.test_text)
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
	self.model:add(nn.SoftMax())
	self.criterion=nn.ClassNLLCriterion()	
	self.label_tensors=torch.Tensor(self.batch_size,1)
end

function Senna:cuda()
	require 'cunn'
	self.model:cuda()
	self.criterion:cuda()
	self.label_tensors:cuda()
end