# Net
## Basic Net
`neural_net.BasicNet` is (supposed to be) a 3-layer neural network for linear regression. It takes the xyz coordinates of two consecutive timepoints of the trajectory and predicts the xyz coordinates of the next timepoint. 
### Issue
The Relu fuction elimates the negative outputs. That would be a problem, since the position vector of the ball needs to be able to have both  negative and postive components. A model that only uses Relu (or leaky Relu for that matter) will never be trained properly. But, I also don't want to use sigmoid fuctions. Hence, we need to think of how else we are going to add non-linearity to the model. 