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
for i=0,103 do
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



net = nn.Sequential()
net:add(nn.SpatialConvolution(1, 6, 5, 5)) -- 1 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU6())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU6())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 400))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU6())                       -- non-linearity 
net:add(nn.Linear(400, 400))
net:add(nn.ReLU6())                       -- non-linearity 
net:add(nn.Linear(400, 104))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems



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
net:zeroGradParameters()
for iter=1,10 do
 for i = 1,#labels do
	pred = net:forward(images[i])
 	err = criterion:forward(pred, labels[i]) 
 	gradCriterion = criterion:backward(pred, labels[i])
 	net:backward(images[i], gradCriterion)
 	net:updateParameters(0.001)
 	net:zeroGradParameters()
 	end
 -- if(iter%1==0) then
 	torch.save('mytrain'..tostring(iter)..'_'..tostring(err)..'.th', net)
 	print(iter)
 	print(err)
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

