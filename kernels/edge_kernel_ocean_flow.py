from typing import Optional
import torch 
from gpytorch.kernels import Kernel 
from gpytorch.constraints import Positive


class DiffusionKernelOceanFlowNonHC(Kernel):
    def __init__(self, eigenpairs): 
        super().__init__()
        self.grad_eigvects, self.curl_eigvects, self.grad_eigvals, self.curl_eigvals = eigenpairs
        # register the raw parameters
        self.register_parameter(
            name='raw_kappa_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_kappa_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        # set the kappa constraints
        self.register_constraint(
            'raw_kappa_down', Positive()
        )
        self.register_constraint(
            'raw_kappa_up', Positive()
        )
        # we do not set the prior on the parameters 

    # set up the actual parameters 
    @property
    def kappa_down(self):
        return self.raw_kappa_down_constraint.transform(self.raw_kappa_down)

    @kappa_down.setter
    def kappa_down(self, value):
        self._set_kappa_down(value)

    def _set_kappa_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_down)
        self.initialize(raw_kappa_down=self.raw_kappa_down_constraint.inverse_transform(value))

    @property
    def kappa_up(self):
        return self.raw_kappa_up_constraint.transform(self.raw_kappa_up)
    
    @kappa_up.setter
    def kappa_up(self, value):
        self._set_kappa_up(value)

    def _set_kappa_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_up)
        self.initialize(raw_kappa_up=self.raw_kappa_up_constraint.inverse_transform(value))
 
    def _eval_covar_matrix(self):
        """Define the full covariance matrix -- full kernel matrix as a property to avoid repeative computation of the kernel matrix"""
        k1 = torch.exp(-self.kappa_down*self.grad_eigvals).squeeze()
        k2 = torch.exp(-self.kappa_down*self.curl_eigvals).squeeze()
        return (k1, k2 )
    
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
            
        (k1, k2) = self._eval_covar_matrix()
        K1 = self.grad_eigvects[x1,:] * k1 @ self.grad_eigvects[x2,:].T
        K2 = self.curl_eigvects[x1,:] * k2 @ self.curl_eigvects[x2,:].T
        K = K1+K2 
        if diag:
            return K.diag()
        else:
            return K 


class DiffusionKernelOceanFlow(Kernel):
    def __init__(self, eigenpairs, kappa_bounds=(1e-5,1e5)): 
        super().__init__()
        self.grad_eigvects, self.curl_eigvects, self.grad_eigvals, self.curl_eigvals = eigenpairs
        # register the raw parameters
        self.register_parameter(
            name='raw_kappa_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_kappa_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_h_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_h_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        # set the kappa constraints
        self.register_constraint(
            'raw_kappa_down', Positive()
        )
        self.register_constraint(
            'raw_kappa_up', Positive()
        )
        self.register_constraint(
            'raw_h_down', Positive()
        )
        self.register_constraint(
            'raw_h_up', Positive()
        )

    # set up the actual parameters 
    @property
    def kappa_down(self):
        return self.raw_kappa_down_constraint.transform(self.raw_kappa_down)

    @kappa_down.setter
    def kappa_down(self, value):
        self._set_kappa_down(value)

    def _set_kappa_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_down)
        self.initialize(raw_kappa_down=self.raw_kappa_down_constraint.inverse_transform(value))

    @property
    def kappa_up(self):
        return self.raw_kappa_up_constraint.transform(self.raw_kappa_up)
    
    @kappa_up.setter
    def kappa_up(self, value):
        self._set_kappa_up(value)

    def _set_kappa_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_up)
        self.initialize(raw_kappa_up=self.raw_kappa_up_constraint.inverse_transform(value))
 
    @property
    def h_down(self):
        return self.raw_h_down_constraint.transform(self.raw_h_down)

    @h_down.setter
    def h_down(self, value):
        self._set_h_down(value)
        
    def _set_h_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_h_down)
        self.initialize(raw_h_down=self.raw_h_down_constraint.inverse_transform(value))
        
    @property
    def h_up(self):
        return self.raw_h_up_constraint.transform(self.raw_h_up)

    @h_up.setter
    def h_up(self, value):
        self._set_h_up(value)

    def _set_h_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_h_up)
        self.initialize(raw_h_up=self.raw_h_up_constraint.inverse_transform(value)) 
 
    def _eval_covar_matrix(self):
        """Define the full covariance matrix -- full kernel matrix as a property to avoid repeative computation of the kernel matrix"""
        k1 = torch.exp(-self.kappa_down*self.grad_eigvals).squeeze()
        k2 = torch.exp(-self.kappa_up*self.curl_eigvals).squeeze()
        return (k1, k2 )
    
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
            
        (k1, k2) = self._eval_covar_matrix()
        K1 = self.grad_eigvects[x1,:] * k1 @ self.grad_eigvects[x2,:].T 
        K2 = self.curl_eigvects[x1,:] * k2 @ self.curl_eigvects[x2,:].T 
        K = self.h_down*K1 + self.h_up*K2 
        if diag:
            return K.diag()
        else:
            return K 


class LaplacianKernelOceanFlow(Kernel):
    def __init__(self, eigenpairs, kappa_bounds=(1e-5,1e5)): 
        super().__init__()
        self.grad_eigvects, self.curl_eigvects, self.grad_eigvals, self.curl_eigvals = eigenpairs
        # register the raw parameters
        self.register_parameter(
            name='raw_kappa_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_kappa_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        # set the kappa constraints
        self.register_constraint(
            'raw_kappa_down', Positive()
        )
        self.register_constraint(
            'raw_kappa_up', Positive()
        )
        # we do not set the prior on the parameters 

    # set up the actual parameters 
    @property
    def kappa_down(self):
        return self.raw_kappa_down_constraint.transform(self.raw_kappa_down)

    @kappa_down.setter
    def kappa_down(self, value):
        self._set_kappa_down(value)

    def _set_kappa_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_down)
        self.initialize(raw_kappa_down=self.raw_kappa_down_constraint.inverse_transform(value))

    @property
    def kappa_up(self):
        return self.raw_kappa_up_constraint.transform(self.raw_kappa_up)
    
    @kappa_up.setter
    def kappa_up(self, value):
        self._set_kappa_up(value)

    def _set_kappa_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_up)
        self.initialize(raw_kappa_up=self.raw_kappa_up_constraint.inverse_transform(value))
 
    def _eval_covar_matrix(self):
        """Define the full covariance matrix -- full kernel matrix as a property to avoid repeative computation of the kernel matrix"""
        k1 = self.kappa_down*1/self.grad_eigvals
        k2 = self.kappa_up*1/self.curl_eigvals
        return (k1, k2 )
    
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
            
        (k1, k2) = self._eval_covar_matrix()
        K1 = self.grad_eigvects[x1,:] * k1 @ self.grad_eigvects[x2,:].T
        K2 = self.curl_eigvects[x1,:] * k2 @ self.curl_eigvects[x2,:].T
        K = K1+K2 
        if diag:
            return K.diag()
        else:
            return K 


class LaplacianKernelOceanFlowNonHC(Kernel):
    def __init__(self, eigenpairs, kappa_bounds=(1e-5,1e5)): 
        super().__init__()
        self.grad_eigvects, self.curl_eigvects, self.grad_eigvals, self.curl_eigvals = eigenpairs
        # register the raw parameters
        self.register_parameter(
            name='raw_kappa_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_kappa_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        # set the kappa constraints
        self.register_constraint(
            'raw_kappa_down', Positive()
        )
        self.register_constraint(
            'raw_kappa_up', Positive()
        )

    # set up the actual parameters 
    @property
    def kappa_down(self):
        return self.raw_kappa_down_constraint.transform(self.raw_kappa_down)

    @kappa_down.setter
    def kappa_down(self, value):
        self._set_kappa_down(value)

    def _set_kappa_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_down)
        self.initialize(raw_kappa_down=self.raw_kappa_down_constraint.inverse_transform(value))

    @property
    def kappa_up(self):
        return self.raw_kappa_up_constraint.transform(self.raw_kappa_up)
    
    @kappa_up.setter
    def kappa_up(self, value):
        self._set_kappa_up(value)

    def _set_kappa_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_up)
        self.initialize(raw_kappa_up=self.raw_kappa_up_constraint.inverse_transform(value))
 
    def _eval_covar_matrix(self):
        """Define the full covariance matrix -- full kernel matrix as a property to avoid repeative computation of the kernel matrix"""
        k1 = 1/self.grad_eigvals
        k2 = 1/self.curl_eigvals
        return (k1, k2 )
    
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
            
        (k1, k2) = self._eval_covar_matrix()
        K1 = self.grad_eigvects[x1,:] * k1 @ self.grad_eigvects[x2,:].T
        K2 = self.curl_eigvects[x1,:] * k2 @ self.curl_eigvects[x2,:].T
        K = K1+K2 
        if diag:
            return K.diag()
        else:
            return K 


class MaternKernelOceanFlow(Kernel):
    def __init__(self, eigenpairs, kappa_bounds=(1e-5,1e5)): 
        super().__init__()
        self.grad_eigvects, self.curl_eigvects, self.grad_eigvals, self.curl_eigvals = eigenpairs
        # register the raw parameters
        self.register_parameter(
            name='raw_kappa_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_kappa_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_mu_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_mu_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_h_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_h_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )

        # set the kappa constraints
        self.register_constraint(
            'raw_kappa_down', Positive()
        )
        self.register_constraint(
            'raw_kappa_up', Positive()
        )
        self.register_constraint(
            'raw_mu_down', Positive()
        )
        self.register_constraint(
            'raw_mu_up', Positive()
        )
        self.register_constraint(
            'raw_h_down', Positive()
        )
        self.register_constraint(
            'raw_h_up', Positive()
        )


    # set up the actual parameters 
    @property
    def kappa_down(self):
        return self.raw_kappa_down_constraint.transform(self.raw_kappa_down)

    @kappa_down.setter
    def kappa_down(self, value):
        self._set_kappa_down(value)

    def _set_kappa_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_down)
        self.initialize(raw_kappa_down=self.raw_kappa_down_constraint.inverse_transform(value))
        
    @property
    def kappa_up(self):
        return self.raw_kappa_up_constraint.transform(self.raw_kappa_up)
    
    @kappa_up.setter
    def kappa_up(self, value):
        self._set_kappa_up(value)

    def _set_kappa_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_up)
        self.initialize(raw_kappa_up=self.raw_kappa_up_constraint.inverse_transform(value))
 
    @property
    def mu_down(self):
        return self.raw_mu_down_constraint.transform(self.raw_mu_down)
    
    @mu_down.setter
    def mu_down(self, value):
        self._set_mu_down(value)

    def _set_mu_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mu_down)
        self.initialize(raw_mu_down=self.raw_mu_down_constraint.inverse_transform(value))

    @property
    def mu_up(self):
        return self.raw_mu_up_constraint.transform(self.raw_mu_up)
    
    @mu_up.setter
    def mu_up(self, value):
        self._set_mu_up(value)

    def _set_mu_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mu_up)
        self.initialize(raw_mu_up=self.raw_mu_up_constraint.inverse_transform(value))

    @property
    def h_down(self):
        return self.raw_h_down_constraint.transform(self.raw_h_down)
    
    @h_down.setter
    def h_down(self, value):
        self._set_h_down(value)

    def _set_h_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_h_down)
        self.initialize(raw_h_down=self.raw_h_down_constraint.inverse_transform(value))

    @property
    def h_up(self):
        return self.raw_h_up_constraint.transform(self.raw_h_up)
    
    @h_up.setter
    def h_up(self, value):
        self._set_h_up(value)

    def _set_h_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_h_up)
        self.initialize(raw_h_up=self.raw_h_up_constraint.inverse_transform(value))


    def _eval_covar_matrix(self):
        """Define the full covariance matrix -- full kernel matrix as a property to avoid repeative computation of the kernel matrix"""
        k1 = torch.pow(2*self.mu_down/self.kappa_down/self.kappa_down + self.grad_eigvals, -self.mu_down) 
        k2 = torch.pow(2*self.mu_up/self.kappa_up/self.kappa_up + self.curl_eigvals, -self.mu_up) 
        return (k1, k2)
    
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
            
        (k1, k2) = self._eval_covar_matrix()
        K1 = self.grad_eigvects[x1,:] * k1 @ self.grad_eigvects[x2,:].T
        K2 = self.curl_eigvects[x1,:] * k2 @ self.curl_eigvects[x2,:].T
        K = self.h_down*K1 + self.h_up*K2 
        if diag:
            return K.diag()
        else:
            return K 


class MaternKernelOceanFlowNonHC(Kernel):
    def __init__(self, eigenpairs): 
        super().__init__()
        self.grad_eigvects, self.curl_eigvects, self.grad_eigvals, self.curl_eigvals = eigenpairs
        # register the raw parameters
        self.register_parameter(
            name='raw_kappa_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_kappa_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_mu_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_mu_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )

        # set the kappa constraints
        self.register_constraint(
            'raw_kappa_down', Positive()
        )
        self.register_constraint(
            'raw_kappa_up', Positive()
        )
        self.register_constraint(
            'raw_mu_down', Positive()
        )
        self.register_constraint(
            'raw_mu_up', Positive()
        )

    # set up the actual parameters 
    @property
    def kappa_down(self):
        return self.raw_kappa_down_constraint.transform(self.raw_kappa_down)

    @kappa_down.setter
    def kappa_down(self, value):
        self._set_kappa_down(value)

    def _set_kappa_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_down)
        self.initialize(raw_kappa_down=self.raw_kappa_down_constraint.inverse_transform(value))
        
    @property
    def kappa_up(self):
        return self.raw_kappa_up_constraint.transform(self.raw_kappa_up)
    
    @kappa_up.setter
    def kappa_up(self, value):
        self._set_kappa_up(value)

    def _set_kappa_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa_up)
        self.initialize(raw_kappa_up=self.raw_kappa_up_constraint.inverse_transform(value))
 
    @property
    def mu_down(self):
        return self.raw_mu_down_constraint.transform(self.raw_mu_down)
    
    @mu_down.setter
    def mu_down(self, value):
        self._set_mu_down(value)

    def _set_mu_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mu_down)
        self.initialize(raw_mu_down=self.raw_mu_down_constraint.inverse_transform(value))

    @property
    def mu_up(self):
        return self.raw_mu_up_constraint.transform(self.raw_mu_up)
    
    @mu_up.setter
    def mu_up(self, value):
        self._set_mu_up(value)

    def _set_mu_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mu_up)
        self.initialize(raw_mu_up=self.raw_mu_up_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        """Define the full covariance matrix -- full kernel matrix as a property to avoid repeative computation of the kernel matrix"""
        k1 = torch.pow(2*self.mu_down/self.kappa_down/self.kappa_down + self.grad_eigvals, -self.mu_down) 
        k2 = torch.pow(2*self.mu_down/self.kappa_down/self.kappa_down + self.curl_eigvals, -self.mu_down) 
        return (k1, k2)
    
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
            
        (k1, k2) = self._eval_covar_matrix()
        K1 = self.grad_eigvects[x1,:] * k1 @ self.grad_eigvects[x2,:].T
        K2 = self.curl_eigvects[x1,:] * k2 @ self.curl_eigvects[x2,:].T
        K = K1+K2 
        if diag:
            return K.diag()
        else:
            return K 