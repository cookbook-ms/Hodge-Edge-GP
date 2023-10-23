from typing import Optional
import torch 
from gpytorch.kernels import Kernel 
from gpytorch.constraints import Positive



class MaternKernelGraph(Kernel):
    def __init__(self, eigenpairs): 
        super().__init__()
        self.eigvals , self.eigvects = eigenpairs
        # register the raw parameters
        self.register_parameter(
            name='raw_kappa', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_mu', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )

        # set the kappa constraints
        self.register_constraint(
            'raw_kappa', Positive()
        )
        self.register_constraint(
            'raw_mu', Positive()
        )

    # set up the actual parameters 
    @property
    def kappa(self):
        return self.raw_kappa_constraint.transform(self.raw_kappa)

    @kappa.setter
    def kappa(self, value):
        self._set_kappa(value)

    def _set_kappa(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa)
        self.initialize(raw_kappa=self.raw_kappa_constraint.inverse_transform(value))
 
    @property
    def mu(self):
        return self.raw_mu_constraint.transform(self.raw_mu)
    
    @mu.setter
    def mu(self, value):
        self._set_mu(value)

    def _set_mu(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mu)
        self.initialize(raw_mu=self.raw_mu_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        """Define the full covariance matrix -- full kernel matrix as a property to avoid repeative computation of the kernel matrix"""
        k1 = torch.pow(2*self.mu/self.kappa/self.kappa + self.eigvals, -self.mu).squeeze() 
        return k1
    
    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()
        
    # define the kernel function 
    def forward(self, x1, x2=None, diag: Optional[bool] = False, **params):
        x1, x2 = x1.long(), x2.long()
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        # compute the kernel matrix
        if x2 is None: 
            x2 = x1
            
        k1 = self._eval_covar_matrix()
        K = self.eigvects[x1,:] * k1 @ self.eigvects[x2,:].T
        if diag:
            return K.diag()
        else:
            return K 



class DiffusionKernelGraph(Kernel):
    def __init__(self, eigenpairs, kappa_bounds=(1e-5,1e5)): 
        super().__init__()
        self.eigvals, self.eigvects = eigenpairs
        # register the raw parameters
        self.register_parameter(
            name='raw_kappa', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        # set the kappa constraints
        self.register_constraint(
            'raw_kappa', Positive()
        )

    # set up the actual parameters 
    @property
    def kappa(self):
        return self.raw_kappa_constraint.transform(self.raw_kappa)

    @kappa.setter
    def kappa(self, value):
        self._set_kappa(value)

    def _set_kappa(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa)
        self.initialize(raw_kappa=self.raw_kappa_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        """Define the full covariance matrix -- full kernel matrix as a property to avoid repeative computation of the kernel matrix"""
        k1 = torch.exp(-self.kappa*self.eigvals).squeeze()
        return k1
    
    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()
        
    # define the kernel function 
    def forward(self, x1, x2=None, diag: Optional[bool] = False, **params):
        x1, x2 = x1.long(), x2.long()
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        # compute the kernel matrix
        if x2 is None: 
            x2 = x1
            
        k1 = self._eval_covar_matrix()
        K = self.eigvects[x1,:] * k1 @ self.eigvects[x2,:].T
        if diag:
            return K.diag()
        else:
            return K 

