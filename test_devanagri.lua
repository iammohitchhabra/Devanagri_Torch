require 'torch'
require 'nn'
require 'image'
require 'nnx'
require 'cutorch'
require 'cunn'
require 'optim'
require 'paths'

correct=0;
wrong=0

labels={}
i=1
for line in io.lines('/home/itachi3/Documents/Research/codes/Database/devanagri/valid/labels.txt') do
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
	images[i]=image.load('/home/itachi3/Documents/Research/codes/Database/devanagri/valid/'..tostring(i-1)..'.png')
	images[i]=image.scale(images[i],32,32)
	images[i]=images[i]:cuda()
	mean = images[i]:mean()
	std=images[i]:std()
	images[i]:add(-mean)
	images[i]:div(std)
	--labels[i]=labels[i]:cuda()
end

model=torch.load('/home/itachi3/Documents/Research/codes/torch/workspace/devnagri/src/mytrain70_3111.028482914.th')
model=model:cuda()
for t=1,#labels do
	pred=model:forward(images[t])
	min,test_label=torch.max(pred,1)
	if(test_label[1]==labels[t]) then
		correct=correct+1
	else
		wrong=wrong+1
	end
end

print(correct)
print('\n')
print(wrong)


