
# from pypointmatcher import pointmatcher, pointmatchersupport
# import wmrde
import os
import torch
from torch.utils.data import DataLoader
# from torchmin import minimize
import gpytorch

from util.util_func import *

from eval.torch_slip_dataset import TorchWMRSlipDataset

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x.float(), train_y.float(), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ApproximateGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 200

slip_dataset_path = '/home/dominic/repos/norlab_WMRD/data/marmotte/ga_hard_snow_25_01_a/slip_dataset_all.pkl'
# train_dataset_path = '/home/dominic/repos/norlab_WMRD/data/husky/vel_mask_array_all.npy'

slip_train_dataset = TorchWMRSlipDataset(slip_dataset_path, gp_state='x')
slip_train_dl = DataLoader(slip_train_dataset)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(slip_train_dataset.x_train, slip_train_dataset.y_train, likelihood)

model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(slip_train_dataset.x_train.float())
    # Calc loss and backprop gradients
    loss = -mll(output, slip_train_dataset.y_train.float())
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

torch.save(model.state_dict(), 'training_results/marmotte/gp_slip/grand_salon_a/model_state.pth')

# slip_test_dataset_path = '/home/dominic/repos/norlab_WMRD/data/marmotte/ga_hard_snow_25_01_b/slip_dataset_all.pkl'
# slip_test_dataset = TorchWMRSlipDataset(slip_test_dataset_path, gp_state='y')
#
# f_preds = model(slip_test_dataset.x_train)
# y_preds = likelihood(model(slip_test_dataset.x_train))
#
# f_mean = f_preds.mean
# f_var = f_preds.variance
# f_covar = f_preds.covariance_matrix
# f_samples = f_preds.sample(sample_shape=torch.Size(1000,))

# # Get into evaluation (predictive posterior) mode
# model.eval()
# likelihood.eval()
#
# # Test points are regularly spaced along [0,1]
# # Make predictions by feeding model through likelihood
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     test_x = torch.linspace(0, 1, 51)
#     observed_pred = likelihood(model(test_x))
#
# with torch.no_grad():
#     # Initialize plot
#     f, ax = plt.subplots(1, 1, figsize=(4, 3))
#
#     # Get upper and lower confidence bounds
#     lower, upper = observed_pred.confidence_region()
#     # Plot training data as black stars
#     ax.plot(slip_train_dataset.x_train.numpy(), slip_train_dataset.y_train.numpy(), 'k*')
#     # Plot predictive means as blue line
#     ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
#     # Shade between the lower and upper confidence bounds
#     ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     ax.set_ylim([-3, 3])
#     ax.legend(['Observed Data', 'Mean', 'Confidence'])


