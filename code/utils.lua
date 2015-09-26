--[[

Utility function used by lua classes.

--]]

require 'torch'

local utils={}

-- Function to check if the input is a valid number
function utils.isNumber(a)
	if tonumber(a) ~= nil then
		return true
	end
	return false
end

-- Function to trim the string
function utils.trim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

-- Function to split a string by given char.
function utils.splitByChar(str,inSplitPattern)
	outResults={}
	local theStart = 1
	local theSplitStart,theSplitEnd=string.find(str,inSplitPattern,theStart)
	while theSplitStart do
		table.insert(outResults,string.sub(str,theStart,theSplitStart-1))
		theStart=theSplitEnd+1
		theSplitStart,theSplitEnd=string.find(str,inSplitPattern,theStart)
	end
	table.insert(outResults,string.sub(str,theStart))
	return outResults
end

-- Function to pad tokens.
function utils.padTokens(tokens,pad)
	local res={}

	-- Append begin tokens
	for i=1,pad do
		table.insert(res,'<bpad-'..i..'>')
	end

	for _,word in ipairs(tokens) do
		table.insert(res,word)
	end

	-- Append end tokens
	for i=1,pad do
		table.insert(res,'<epad-'..i..'>')
	end

	return res
end

-- Function to get all ngrams
function utils.getNgrams(doc,n,pad)
	local res={}
	local tokens=utils.padTokens(utils.splitByChar(doc,' '),pad)
	for i=1,(#tokens-n+1) do
		local word=''
		for j=i,(i+(n-1)) do
			word=word..tokens[j]..' '
		end
		word=utils.trim(word)
		table.insert(res,word)
	end
	return res
end

-- Function to process a sentence to build vocab
function utils.processSentence(config,sentence)
	local pad=(config.wwin/2)
	for _,word in ipairs(utils.getNgrams(sentence,1,pad)) do
		config.total_count=config.total_count+1

		word=utils.splitByChar(word,'%$%$%$')[1]

		if config.to_lower==1 then
			word=word:lower()
		end

		-- Fill word vocab.
		if config.vocab[word]==nil then
			config.vocab[word]=1
		else
			config.vocab[word]=config.vocab[word]+1
		end
	end
	config.corpus_size=config.corpus_size+1
end

-- Function to get word tensor
function utils.getWordTensor(config,words)
	local wordTensor=torch.Tensor(#words)
	for i,word in ipairs(words) do
		if config.word2index[word]==nil then
			wordTensor[i]=config.word2index['<UK>']
		else
			wordTensor[i]=config.word2index[word]
		end
	end
	return wordTensor
end

-- Function to get input tensors.
function utils.getFullInputTensors(config,sentence)
	local tensors={}
	local pad=(config.wwin/2)
	local words=utils.getNgrams(sentence,1,pad)
	local wordTensor=utils.getWordTensor(config,words)
	for i,word in ipairs(words) do
		-- Get word and label
		local content=utils.splitByChar(word,'%$%$%$')
		word=content[1]
		local label=nil
		if content[2]==nil then 
			label=1
		elseif content[2]=='I' then
			label=2
		else
			label=1
		end
		if config.to_lower==1 then
			word=word:lower()
		end
		local labelTensor=torch.Tensor(1):fill(label)
		local posTensor=torch.Tensor(#words,1)
		for j=1,#words do
			posTensor[j]=math.abs(j-i)
		end
		table.insert(tensors,{{wordTensor,posTensor},labelTensor})
	end
	return tensors
end

-- Function to build vocabulary from the corpus
function utils.buildVocab(config)
	print('Building vocabulary...')
	local start=sys.clock()
	local fptr=io.open(config.train_file,'r')
	
	-- Fill the vocabulary frequency map
	config.total_count=0
	config.corpus_size=0
	config.corpus_text={}
	while true do
		local pid=fptr:read()
		if pid==nil then
			break
		end
		local abstract=fptr:read()
		utils.processSentence(config,abstract)
		table.insert(config.corpus_text,abstract)
		local noOfSent=tonumber(fptr:read())
		for i=1,noOfSent do
			local sentence=fptr:read()
			utils.processSentence(config,sentence)
			table.insert(config.corpus_text,sentence)
		end
	end
	fptr.close()

	-- Discard the words that doesn't meet minimum frequency and create indices.
	for word,count in pairs(config.vocab) do
		if count<config.min_freq then
			config.vocab[word]=nil
		else
			config.index2word[#config.index2word+1]=word
			config.word2index[word]=#config.index2word
		end
	end

	-- Add unknown word
	config.vocab['<UK>']=1
	config.index2word[#config.index2word+1]='<UK>'
	config.word2index['<UK>']=#config.index2word
	config.vocab_size= #config.index2word

	print(string.format("%d words, %d documents processed in %.2f seconds.",config.total_count,config.corpus_size,sys.clock()-start))
	print(string.format("Vocab size after eliminating words occuring less than %d times: %d",config.min_freq,config.vocab_size))
end

-- Function to load dev corpus
function utils.loadDevCorpus(config)
	local fptr=io.open(config.dev_file,'r')
	config.dev_text={}
	while true do
		local pid=fptr:read()
		if pid==nil then
			break
		end
		local abstract=fptr:read()
		table.insert(config.dev_text,abstract)
		local noOfSent=tonumber(fptr:read())
		for i=1,noOfSent do
			local sentence=fptr:read()
			table.insert(config.dev_text,sentence)
		end
	end
	fptr.close()
end

-- Function to load test corpus
function utils.loadTestCorpus(config)
	local fptr=io.open(config.test_file,'r')
	config.test_text={}
	while true do
		local pid=fptr:read()
		if pid==nil then
			break
		end
		local abstract=fptr:read()
		utils.processSentence(config,abstract)
		local noOfSent=tonumber(fptr:read())
		for i=1,noOfSent do
			local sentence=fptr:read()
			table.insert(config.test_text,sentence)
		end
	end
	fptr:close()
end

-- Function to find predicted class
function utils.argmax(v)
	local idx=1
	local max=v[1]
	for i=2,v:size(1) do
		if v[i]>max then
			max=v[i]
			idx=i
		end
	end
	return idx
end

-- Function to find accuracy
function utils.accuracy(pred,gold)
	return torch.eq(pred,gold):sum()/pred:size(1)
end

return utils