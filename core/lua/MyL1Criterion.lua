MyL1Criterion, parent = torch.class('nn.MyL1Criterion', 'nn.Criterion')

function MyL1Criterion:__init()
   parent.__init(self)
end

function MyL1Criterion:updateOutput(input, target)
   -- loss is the Euclidean distance between predicted and ground truth coordinate, mean calculated over batch
   self.output = torch.mean(torch.norm(input - target, 2, 2))
   return self.output
end

function MyL1Criterion:updateGradInput(input, target)
   -- gradients are the difference of predicted and ground truth coordinate divided (scaled) by the Euclidean distance
   local dists = torch.norm(input - target, 2, 2)
   dists = torch.expand(dists, dists:size(1), 3)
   self.gradInput = torch.cdiv(input-target,dists)
   self.gradInput = torch.div(self.gradInput, dists:size(1))
   return self.gradInput
end
