require "nn"
require "cunn"
require 'optim'
require 'cudnn'

mean = {45} -- value substracted from input for normalization (reprojection errors range from 0 to 100)

-- general parameters
storeCounter = 0 -- counts parameter updates

-- parameters of end to end training
storeIntervalE2E = 1000 -- storing snapshot after x updates
clampE2E = 0.1 -- maximum gradient magnitude
lrInitE2E = 0.0000001 -- (initial) learning rate of end-to-end training
momentumE2E = 0.9 -- SGD momentum
sFileE2E = 'score_model_softam_endtoend.net' -- output file name of end-to-end training
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

