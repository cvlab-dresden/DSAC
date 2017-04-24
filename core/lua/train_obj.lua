require "nn"
require "cunn"
require 'optim'
require 'cudnn'

-- general parameters
storeCounter = 0 -- counts parameter updates
batchSize = 1600 -- how many patches to process simultaneously, determined by GPU memory

-- parameters of pretraining
storeIntervalPre = 1000 -- storing snapshot after x updates
lrInitPre = 0.0001 -- (initial) learning rate of componentwise pre-training
lrInterval = 50000 -- cutting learning rate in half after x updates
oFilePre = 'obj_model_init.net' -- output file name of componentwise pre-training

-- parameters of end-to-end training
storeIntervalE2E = 1000 -- storing snapshot after x updates
lrInitE2E = 0.0001 -- (initial) learning rate of end-to-end training
lrDecayE2E = 0.1 -- learning rate decay
momentumE2E = 0.9 -- SGD momentum
clampE2E = 0.1 -- maximum gradient magnitude
oFileE2E = 'obj_model_endtoend.net' -- output file name of end-to-end training

mean = {127, 127, 127} -- value substracted from each RGB input patch for normalization

dofile('MyL1Criterion.lua') -- custom L1 loss for coordinates

-- loads a network from file f, also sets number of input channels to c and input patch size to s (assumes square patches)
function loadModel(f, c, s)

  print('TORCH: Loading network from file: ' .. f)

  channels = c
  inputSize = s

  model = torch.load(f)
  model = model:cuda()
  cudnn.convert(model, cudnn)

  model:evaluate()

  criterion = nn.MyL1Criterion()
  criterion = criterion:cuda()

  params, gradParams = model:getParameters()
  optimState = {learningRate = lrInitE2E, momentum = momentumE2E}
end

-- constructs a new network, also sets number of input channels to c and input patch size to s (assumes square patches)
function constructModel(c, s)

  channels = c
  inputSize = s

  print('TORCH: Creating network.')
  --42x42
  model = nn.Sequential()
  model:add(nn.SpatialConvolution(channels, 64, 3, 3, 1, 1, 0, 0))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(64, 64, 3, 3, 2, 2, 1, 1))
  model:add(nn.ReLU()) 
  --20x20
  model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(128, 128, 3, 3, 2, 2, 1, 1))
  model:add(nn.ReLU()) 
  --10x10
  model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(256, 256, 3, 3, 2, 2, 1, 1))
  model:add(nn.ReLU()) 
  --5x5
  model:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
  model:add(nn.ReLU()) 
  model:add(nn.SpatialConvolution(512, 512, 3, 3, 2, 2, 0, 0))
  model:add(nn.ReLU()) 
  --2x2
  model:add(nn.View(2*2*512))

  model:add(nn.Linear(2*2*512, 4096))
  model:add(nn.ReLU())

  model:add(nn.Linear(4096, 4096))
  model:add(nn.ReLU())

  model:add(nn.Linear(4096, 3))

  criterion = nn.MyL1Criterion()

  model = model:cuda()
  cudnn.convert(model, cudnn)

  model:evaluate()

  criterion = criterion:cuda()

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
-- returns list of 'count' coordinate predictions 
function forward(count, data)

  local input = torch.FloatTensor(data):reshape(count, channels, inputSize, inputSize);

  -- normalize data
  for c=1,channels do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  print('TORCH: Doing a forward pass for ' .. count .. ' patches.')

  local batchCount = math.ceil(count / batchSize)
  local results = torch.Tensor(count, 3)

  -- batch-wise forward
  for b=1,batchCount do
      -- assemble batch
      local batchLower = (b-1) * batchSize + 1
      local batchUpper = math.min(b * batchSize, count)

      local batchInput = input[{{batchLower, batchUpper},{},{},{}}]
      batchInput = batchInput:cuda()

      -- forward
      local batchResults = model:forward(batchInput)

      -- write results
      results[{{batchLower, batchUpper}, {}}]:copy(batchResults)
  end

  -- write results to list to be passed back to c++
  results = results:double()

  local resultsR = {}
  for i = 1,results:size(1) do
    for j = 1,3 do
      local idx = (i-1) * 3 + j
      resultsR[idx] = results[{i, j}]
    end
  end

  return resultsR
end

-- performs a backward pass of the network with pre-calculated loss and gradients (for end-to-end training)
-- 'data' consists of 'count' square patches with 'channels' channels and of side length 'inputSize' pixels
-- 'loss' is the loss as measured externally, e.g. the pose loss at the end of the camera localization pipeline
-- 'gradients' as calculated externally. e.g. by backprob through the camera localization pipeline, assumes 'count' x 3 gradients
function backward(count, loss, data, gradients)

  print('TORCH: Doing a backward pass with ' .. count .. ' patches.')
  local input = torch.FloatTensor(data):reshape(count, channels, inputSize, inputSize);
  local dloss_dpred = torch.FloatTensor(gradients):reshape(count, 3);

  -- clamp gradients to deal with high variance
  dloss_dpred:clamp(-clampE2E,clampE2E)

  -- normalize data
  for c=1,channels do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  gradParams:zero()

  local function feval(params) -- called by the opt package

    local batchCount = math.ceil(count / batchSize)

    -- batch-wise backward
    for b=1,batchCount do
        -- assemble batch
        local batchLower = (b-1) * batchSize + 1
        local batchUpper = math.min(b * batchSize, count)

        local batchInput = input[{{batchLower, batchUpper},{},{},{}}]
        local batchGradients = dloss_dpred[{{batchLower, batchUpper},{}}]

        batchInput = batchInput:cuda()
        batchGradients = batchGradients:cuda()

        -- because we process batch-wise we have to call forward again for this particular batch
        local batchResults = model:forward(batchInput)
        -- backward pass
        model:backward(batchInput, batchGradients)
    end

    return loss,gradParams
  end
  optim.sgd(feval, params, optimState)

  storeCounter = storeCounter + 1

  if (storeCounter % storeIntervalE2E) == 0 then
    print('TORCH: Storing a snapshot of the network.')
    model:clearState()
    torch.save(oFileE2E, model)
  end
    
    optimState.learningRate = lrInitE2E / (1 + lrDecayE2E * storeCounter)
    print('TORCH: Updating learning rate. Is now: ' .. optimState.learningRate)

end

-- performs a full training step with immediate loss function (componentwise pre-training)
-- no internal batch support, provide just as many patches as fit to your GPU
-- 'data' consists of 'count' square patches with 'channels' channels and of side length 'inputSize' pixels
-- 'labels' are the ground truth coordinates for the patches in 'data', 'count' x 3 labels
-- returns the mean loss over the input patches
function train(count, data, labels)
  print('TORCH: Doing a training pass with ' .. count .. ' patches.')

  local input = torch.FloatTensor(data):reshape(count, channels, inputSize, inputSize);
  local output = torch.FloatTensor(labels):reshape(count, 3);
  
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
    torch.save(oFilePre, model)
  end

  if (storeCounter % lrInterval) == 0 then
    print('TORCH: Cutting learningrate by half. Is now: ' .. optimState.learningRate)
    optimState.learningRate = optimState.learningRate * 0.5
  end

  return loss
end

-- calculates the mean (internal) loss for a set of patches (used for validation in componentwise pre-training)
-- 'pred' are predicted coordinates, 'count' x 3 predictions
-- 'labels' are ground truth coordinates, 'count' x 3 labels
function getLoss(count, pred, labels)
  print('TORCH: Calculating loss of ' .. count .. ' predictions.')

  local input = torch.FloatTensor(pred):reshape(count, 3);
  local output = torch.FloatTensor(labels):reshape(count, 3);
  
  input = input:cuda()
  output = output:cuda()

  return criterion:forward(input, output)
end
