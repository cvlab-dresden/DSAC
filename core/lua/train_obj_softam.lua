require "nn"
require "cunn"
require 'optim'
require 'cudnn'

-- general parameters
storeCounter = 0 -- counts parameter updates
batchSize = 1600 -- how many patches to process simultaneously, determined by GPU memory

-- parameters of end-to-end training
storeIntervalE2E = 1000 -- storing snapshot after x updates
lrInitE2E = 0.00001 -- learning rate of end-to-end training
momentumE2E = 0.9 -- SGD momentum
clampE2E = 0.1 -- maximum gradient magnitude
oFileE2E = 'obj_model_softam_endtoend.net'

mean = {127, 127, 127} -- value substracted from each RGB input patch for normalization

-- loads a network from file f, also sets number of input channels to c and input patch size to s (assumes square patches)
function loadModel(f, c, s)

  print('TORCH: Loading network from file: ' .. f)

  channels = c
  inputSize = s

  model = torch.load(f)
  model = model:cuda()
  cudnn.convert(model, cudnn)

  model:evaluate()

  params, gradParams = model:getParameters()
  optimState = {learningRate = lrInitE2E, momentum = momentumE2E}
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
    

end
