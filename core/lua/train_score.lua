require "nn"
require "cunn"
require 'optim'
require 'cudnn'

mean = {45} -- value substracted from input for normalization (reprojection errors range from 0 to 100)

-- general parameters
storeCounter = 0 -- counts parameter updates

-- parameters of pretraining
storeIntervalPre = 100 -- storing snapshot after x updates
lrIntervalPre = 5000 -- cutting learning rate in half after x updates
lrInitPre = 0.0001 -- (initial) learning rate of componentwise pre-training
sFilePre = 'score_model_init.net' -- output file name of componentwise pre-training

-- parameters of end-to-end training
storeIntervalE2E = 1000 -- storing snapshot after x updates
clampE2E = 0.1 -- maximum gradient magnitude
lrInitE2E = 0.0000001 -- (initial) learning rate of end-to-end training
momentumE2E = 0.9 -- SGD momentum
sFileE2E = 'score_model_endtoend.net' -- output file name of end-to-end training
lrIntervalE2E = 10000 -- cutting learning rate in half after x updates

-- loads a network from file f, also sets number of input channels to c and input patch size to s (assumes square patches)
function loadModel(f, c, s)

  print('TORCH: Loading network from file: ' .. f)

  channels = c
  inputSize = s

  model = torch.load(f)
  cudnn.convert(model, cudnn)

  model:evaluate()

  params, gradParams = model:getParameters()
  optimState = {learningRate = lrInitE2E, momentum=momentumE2E}

  criterion = nn.AbsCriterion()
  criterion = criterion:cuda()
end

-- constructs a new network, also sets number of input channels to c and input patch size to s (assumes square patches)
function constructModel(c, s)

  channels = c
  inputSize = s

  print('TORCH: Creating network.')

  --40x40
  model = nn.Sequential()
  model:add(nn.SpatialConvolution(channels, 32, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(32, 32, 3, 3, 2, 2, 1, 1))
  model:add(nn.ReLU()) 
  --20x20
  model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(64, 64, 3, 3, 2, 2, 1, 1))
  model:add(nn.ReLU()) 
  --10x10
  model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(128, 128, 3, 3, 2, 2, 1, 1))
  model:add(nn.ReLU()) 
  --5x5
  model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(256, 256, 3, 3, 2, 2, 0, 0))
  model:add(nn.ReLU()) 
  --2x2
  model:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 2, 2, 1, 1))
  model:add(nn.ReLU()) 
  --1x1
  model:add(nn.View(512))

  model:add(nn.Linear(512, 1024))
  model:add(nn.ReLU())

  model:add(nn.Linear(1024, 1024))
  model:add(nn.ReLU())

  model:add(nn.Linear(1024, 1))

  criterion = nn.AbsCriterion()

  model = model:cuda()
  cudnn.convert(model, cudnn)
  criterion = criterion:cuda()

  model:evaluate()

  params, gradParams = model:getParameters()
  optimState = {learningRate = lrInitPre}
end

function setEvaluate()
    model:evaluate()
    print('TORCH: Set model to evaluation mode.')
end

function setTraining()
    model:training()
    print('TORCH: Set model to training mode.')
end

-- performs a forward pass of the network
-- 'data' consists of 'count' square patches with 'channels' channels and of side length 'inputSize' pixels
-- returns list of 'count' scores
function forward(count, data)
  print('TORCH: Doing a forward pass for ' .. count .. ' images.')
  local input = torch.FloatTensor(data):reshape(count, channels, inputSize, inputSize);
  input = input:cuda()

  -- normalize data
  for c=1,channels do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  -- reset network parameter gradients for asynchronous backward call
  gradParams:zero() 
  -- forward pass
  local results = model:forward(input) 

  -- write results to list to be passed back to c++
  local r = {}
  for i = 1,results:size(1) do
    if count == 1 then
      r[i] = results[{i}]
    else
      r[i] = results[{i, 1}]
    end
  end

  return unpack(r)
end

-- calculates the mean (internal) loss for a set of scores (used for validation in componentwise pre-training)
-- 'pred' are predicted scores, 'count' x 3 predictions
-- 'labels' are ground truth scores, 'count' x 3 labels
function getLoss(count, pred, labels)
  print('TORCH: Calculating loss of ' .. count .. ' predictions.')

  local input = torch.FloatTensor(pred);
  local output = torch.FloatTensor(labels);
  
  input = input:cuda()
  output = output:cuda()

  return criterion:forward(input, output)
end

-- performs a full training step with immediate loss function (componentwise pre-training)
-- 'data' consists of 'count' square patches with 'channels' channels and of side length 'inputSize' pixels
-- 'labels' are the ground truth scores for the patches in 'data', 'count' x 3 labels
-- returns the mean loss over the input patches
function train(count, data, labels)
  print('TORCH: Doing a training pass with ' .. count .. ' images.')
  local input = torch.FloatTensor(data):reshape(count, channels, inputSize, inputSize);
  local output = torch.FloatTensor(labels);
  
  input = input:cuda()
  output = output:cuda()

  -- normalize data
  for c=1,channels do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  local loss = 0

  local function feval(params)
    gradParams:zero()

    -- forward pass
    local pred = model:forward(input)
    -- measure loss
    loss = criterion:forward(pred, output)
    -- loss gradients
    local dloss_dpred = criterion:backward(pred, output)
    -- backward pass
    model:backward(input, dloss_dpred)

    return loss,gradParams
  end
  optim.adam(feval, params, optimState)

  storeCounter = storeCounter + 1

  if (storeCounter % storeIntervalPre) == 0 then
    print('TORCH: Storing a snapshot of the network.')
    model:clearState()
    torch.save(sFilePre, model)
  end

  if (storeCounter % lrIntervalPre) == 0 then
    print('TORCH: Cutting learningrate by half. Is now: ' .. optimState.learningRate)
    optimState.learningRate = optimState.learningRate * 0.5
  end

  return loss
end

-- performs a backward pass of the network with pre-calculated loss and gradients (for end-to-end training)
-- 'data' consists of 'count' square patches with 'channels' channels and of side length 'inputSize' pixels
-- 'outputGradients' as calculated externally. e.g. by backprob through the camera localization pipeline, assumes 'count' gradients
-- returns gradients on the inputs, i.e. a list of 'inputSize'*'inputSize' gradients
function backward(count, data, outputGradients)
  print('TORCH: Doing a backward pass for ' .. count .. ' images.')

  local input = torch.FloatTensor(data):reshape(count, channels, inputSize, inputSize);
  input = input:cuda()

  local gradOutput = torch.FloatTensor(outputGradients):reshape(count, 1);
  gradOutput = gradOutput:cuda()

  -- normalize data
  for c=1,channels do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  -- clamp gradients to deal with high variance
  gradOutput:clamp(-clampE2E,clampE2E)

  -- backward pass 
  local gradInput = model:backward(input, gradOutput)

  local function feval(params) -- called by the opt package
    return 0, gradParams
  end
  optim.sgd(feval, params, optimState)

  storeCounter = storeCounter + 1

  if (storeCounter % storeIntervalE2E) == 0 then
    print('TORCH: Storing a snapshot of the network.')
    model:clearState()
    torch.save(sFileE2E, model)
  end

  if (storeCounter % lrIntervalE2E) == 0 then
    print('TORCH: Cutting learningrate by half. Is now: ' .. optimState.learningRate)
    optimState.learningRate = optimState.learningRate * 0.5
  end

  -- write input gradients to list to be passed back to c++
  gradInput = gradInput:double()

  local gradInputR = {}
  for c = 1,count do
    for x = 1,inputSize do
      for y = 1,inputSize do
         local idx = (c-1) * inputSize * inputSize + (x-1) * inputSize + y
         gradInputR[idx] = gradInput[{c, 1, y, x}]
      end
    end
  end

  return gradInputR
end

