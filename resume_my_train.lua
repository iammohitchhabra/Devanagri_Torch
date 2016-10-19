require 'torch'
require 'nn'
require 'image'
require 'nnx'
require 'cutorch'
require 'cunn'
require 'optim'
require 'paths'


batchSize=5




torch.setdefaulttensortype('torch.FloatTensor')

-- Step1) Define the classes
classes={}
for i=1,103 do
	classes[i]=i
end

-- Step2) Define the labels.
-- Make the the path a cmd variable for better implementation.

labels={}
i=1
for line in io.lines('/home/itachi3/Documents/Research/codes/Database/devanagri/train/labels.txt') do
	labels[i]=tonumber(line)
	i=i+1
end

images={}
-- for i=1,#labels do
-- 	image[i]=image.load('/home/itachi3/Documents/Research/codes/Database/devanagri/train/'..tostring(i)..'.png')
-- end
-- image.display{image=images[50],legend='first'}
-- for i=1,17205 do
-- 	table.insert(images,image.load('/home/itachi3/Documents/Research/codes/Database/devanagri/train/'..tostring(i)..'.png'))
-- end
for i=1,#labels do
	images[i]=image.load('/home/itachi3/Documents/Research/codes/Database/devanagri/train/'..tostring(i-1)..'.png')
	images[i]=image.scale(images[i],32,32)
	images[i]=images[i]:cuda()
	mean = images[i]:mean()
	std=images[i]:std()
	images[i]:add(-mean)
	images[i]:div(std)
	--labels[i]=labels[i]:cuda()
end


net=torch.load('/home/itachi3/Documents/Research/codes/torch/workspace/devnagri/src/mytrain75_4504.284570694.th')

net=net:cuda()

-- trainset={data,labels}

criterion = nn.ClassNLLCriterion()
criterion=criterion:cuda()
-- trainset.data = trainset.data:cuda()
-- trainset.label = trainset.label:cuda()

-- trainer = nn.StochasticGradient(net, criterion)
-- trainer.learningRate = 0.001
-- trainer.maxIteration = 5 -- just do 5 epochs of training.
-- trainer:train(trainset)

print('Lenet5\n' .. net:__tostring());
sumerr=0
for iter=1,500 do
 for i = 1,#labels do
	pred = net:forward(images[i])
 	err = criterion:forward(pred, labels[i]) 
 	sumerr=sumerr+math.abs(err)
	gradCriterion = criterion:backward(pred, labels[i])
 	net:backward(images[i], gradCriterion)
 	net:updateParameters(0.001)
 	net:zeroGradParameters()
 	end
 -- if(iter%5==0) then
 	torch.save('mytrain'..tostring(iter)..'_'..tostring(sumerr)..'.th', net)
 	print(iter)
 	print(sumerr)
 	sumerr=0
 -- end
end















-- for previously trained network model=torch.load(opt.network)


-- input=torch.rand(1,32,32)
-- input=input:cuda()
-- output=net:forward(input)
-- print(output)


-- im=image.scale(images[1],32,32)
-- image.display{im,legend='Scaled image'}
-- criterion=nn.ClassNLLCriterion()


-- mean={}
-- std={}
-- im={}
-- output={}

-- --For some reason these statements if included in the sample for loop, it gives index error
-- for t=1,batchSize do
-- 	im[t]=image.scale(images[t],32,32)
-- end



-- for sample=1,batchSize do
	
-- 	im[sample]=im[sample]:cuda()
-- 	mean = im[sample]:mean()
-- 	std=im[sample]:std()
-- 	im[sample]:add(-mean)
-- 	im[sample]:div(std)
-- 	net:zeroGradParameters()
-- 	output[sample]=net:forward(im[sample])

-- 	criterion=nn.ClassNLLCriterion()
--end




-- image.display{image=images[1],legend='Normalized Image'}

-- images:normalizeGlobal(mean,std)

-- trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
-- testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

