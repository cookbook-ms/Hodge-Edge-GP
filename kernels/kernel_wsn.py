from typing import Optional
import torch 
from gpytorch.kernels import Kernel 
from gpytorch.constraints import Positive


class HodgeDiffusionKernelNonHC(Kernel):
    def __init__(self, incidence_matrices): 
        super().__init__()
        self.sc_order = 1
        self.b1 = incidence_matrices
        self.L0 = self.b1 @ self.b1.T
        self.L1 = self.b1.T @ self.b1
        self.eigvals0, self.eigvecs0 = torch.linalg.eigh(self.L0)
        self.eigvals1, self.eigvecs1 = torch.linalg.eigh(self.L1)
            
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

        self.register_parameter(
            name='raw_gamma_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_gamma_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        # set the gamma constraints
        self.register_constraint(
            'raw_gamma_down', Positive()
        )
        self.register_constraint(
            'raw_gamma_up', Positive()
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
    def gamma_down(self):
        return self.raw_gamma_down_constraint.transform(self.raw_gamma_down)
    
    @gamma_down.setter
    def gamma_down(self, value):
        self._set_gamma_down(value)

    def _set_gamma_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma_down)
        self.initialize(raw_gamma_down=self.raw_gamma_down_constraint.inverse_transform(value))

    @property
    def gamma_up(self):
        return self.raw_gamma_up_constraint.transform(self.raw_gamma_up)
    
    @gamma_up.setter
    def gamma_up(self, value):
        self._set_gamma_up(value)

    def _set_gamma_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma_up)
        self.initialize(raw_gamma_up=self.raw_gamma_up_constraint.inverse_transform(value))
 
    def _eval_covar_matrix(self):
        if self.sc_order == 1:
            k0 = torch.exp(-self.kappa_down*self.eigvals0).squeeze()
            k1 = torch.exp(-self.kappa_up*self.eigvals1).squeeze()
            return k0, k1
    
    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()
        
    # define the kernel function 
    def forward(self, x1, x2=None,diag: Optional[bool] = False,  **params):
        x1, x2 = x1.long(), x2.long()
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        # compute the kernel matrix
        if x2 is None: 
            x2 = x1
            
        if self.sc_order == 1:
            k0, k1 = self._eval_covar_matrix()            
            K0 = self.gamma_down*self.eigvecs0*k0@self.eigvecs0.T
            K1 = self.gamma_up*self.eigvecs1*k1@self.eigvecs1.T 
            KK = torch.block_diag(K0, K1)
            K = KK[x1,:][:,x2]

        if diag:
            return K.diag()
        else:
            return K 
        

class HodgeDiffusionKernel(Kernel):
    def __init__(self, incidence_matrices): 
        super().__init__()
        self.sc_order = 1
        self.b1 = incidence_matrices
        self.L0 = self.b1 @ self.b1.T
        self.L1 = self.b1.T @ self.b1
        self.eigvals0, self.eigvecs0 = torch.linalg.eigh(self.L0)
            
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

        self.register_parameter(
            name='raw_gamma_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_gamma_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        # set the gamma constraints
        self.register_constraint(
            'raw_gamma_down', Positive()
        )
        self.register_constraint(
            'raw_gamma_up', Positive()
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
    def gamma_down(self):
        return self.raw_gamma_down_constraint.transform(self.raw_gamma_down)
    
    @gamma_down.setter
    def gamma_down(self, value):
        self._set_gamma_down(value)

    def _set_gamma_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma_down)
        self.initialize(raw_gamma_down=self.raw_gamma_down_constraint.inverse_transform(value))

    @property
    def gamma_up(self):
        return self.raw_gamma_up_constraint.transform(self.raw_gamma_up)
    
    @gamma_up.setter
    def gamma_up(self, value):
        self._set_gamma_up(value)

    def _set_gamma_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma_up)
        self.initialize(raw_gamma_up=self.raw_gamma_up_constraint.inverse_transform(value))
 
    def _eval_covar_matrix(self):
        if self.sc_order == 1:
            k = torch.exp(-self.kappa_down*self.eigvals0).squeeze()
            return k 

    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()
        
    # define the kernel function 
    def forward(self, x1, x2=None,diag: Optional[bool] = False,  **params):
        x1, x2 = x1.long(), x2.long()
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        # compute the kernel matrix
        if x2 is None: 
            x2 = x1

        if self.sc_order == 1:
            k = self._eval_covar_matrix()            
            K0 = self.gamma_down*self.eigvecs0*k@self.eigvecs0.T
            K1 = self.gamma_up*self.b1.T@K0@self.b1
            KK = torch.block_diag(K0, K1)
            K = KK[x1,:][:,x2]

        if diag:
            return K.diag()
        else:
            return K 


class HodgeMaternKernel(Kernel):
    def __init__(self, incidence_matrices): 
        super().__init__()
        self.sc_order = 1
        self.b1 = incidence_matrices
        self.L0 = self.b1 @ self.b1.T
        self.L1 = self.b1.T @ self.b1
        # use node laplacian 
        self.eigvals0, self.eigvecs0 = torch.linalg.eigh(self.L0)
            
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

        self.register_parameter(
            name='raw_gamma_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_gamma_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        # set the gamma constraints
        self.register_constraint(
            'raw_gamma_down', Positive()
        )
        self.register_constraint(
            'raw_gamma_up', Positive()
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
    def gamma_down(self):
        return self.raw_gamma_down_constraint.transform(self.raw_gamma_down)
    
    @gamma_down.setter
    def gamma_down(self, value):
        self._set_gamma_down(value)

    def _set_gamma_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma_down)
        self.initialize(raw_gamma_down=self.raw_gamma_down_constraint.inverse_transform(value))

    @property
    def gamma_up(self):
        return self.raw_gamma_up_constraint.transform(self.raw_gamma_up)
    
    @gamma_up.setter
    def gamma_up(self, value):
        self._set_gamma_up(value)

    def _set_gamma_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma_up)
        self.initialize(raw_gamma_up=self.raw_gamma_up_constraint.inverse_transform(value))
 
    def _eval_covar_matrix(self):
        if self.sc_order == 1:
            k = torch.pow(2*self.kappa_up/self.kappa_down/self.kappa_down + self.eigvals0, -self.kappa_up)
            return k 
    
    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()
        
    # define the kernel function 
    def forward(self, x1, x2=None,diag: Optional[bool] = False,  **params):
        x1, x2 = x1.long(), x2.long()
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        # compute the kernel matrix
        if x2 is None: 
            x2 = x1
            
        if self.sc_order == 1:
            k = self._eval_covar_matrix()            
            K0 = self.gamma_down*self.eigvecs0*k@self.eigvecs0.T
            K1 = self.gamma_up*self.b1.T@K0@self.b1
            KK = torch.block_diag(K0, K1)
            K = KK[x1,:][:,x2]

        if diag:
            return K.diag()
        else:
            return K 


class HodgeMaternKernelNonHC(Kernel):
    def __init__(self, incidence_matrices): 
        super().__init__()

        self.sc_order = 1
        self.b1 = incidence_matrices
        self.L0 = self.b1 @ self.b1.T
        self.L1 = self.b1.T @ self.b1
        # use node laplacian 
        self.eigvals0, self.eigvecs0 = torch.linalg.eigh(self.L0)
        self.eigvals1, self.eigvecs1 = torch.linalg.eigh(self.L1)
            
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

        self.register_parameter(
            name='raw_mu_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_mu_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_constraint(
            'raw_mu_down', Positive()
        )
        self.register_constraint(
            'raw_mu_up', Positive()
        )

        self.register_parameter(
            name='raw_gamma_down', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        self.register_parameter(
            name='raw_gamma_up', parameter=torch.nn.Parameter(torch.zeros(1,1))
        )
        # set the gamma constraints
        self.register_constraint(
            'raw_gamma_down', Positive()
        )
        self.register_constraint(
            'raw_gamma_up', Positive()
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
    def gamma_down(self):
        return self.raw_gamma_down_constraint.transform(self.raw_gamma_down)
    
    @gamma_down.setter
    def gamma_down(self, value):
        self._set_gamma_down(value)

    def _set_gamma_down(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma_down)
        self.initialize(raw_gamma_down=self.raw_gamma_down_constraint.inverse_transform(value))

    @property
    def gamma_up(self):
        return self.raw_gamma_up_constraint.transform(self.raw_gamma_up)
    
    @gamma_up.setter
    def gamma_up(self, value):
        self._set_gamma_up(value)

    def _set_gamma_up(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gamma_up)
        self.initialize(raw_gamma_up=self.raw_gamma_up_constraint.inverse_transform(value))

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
        if self.sc_order == 1:
            k0 = torch.pow(2*self.mu_down/self.kappa_down/self.kappa_down + self.eigvals0, -self.mu_down).squeeze()
            k1 = torch.pow(2*self.mu_up/self.kappa_up/self.kappa_up + self.eigvals1, -self.mu_up).squeeze()
            return k0, k1
    
    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()
        
    # define the kernel function 
    def forward(self, x1, x2=None,diag: Optional[bool] = False,  **params):
        x1, x2 = x1.long(), x2.long()
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        # compute the kernel matrix
        if x2 is None: 
            x2 = x1

        if self.sc_order == 1:
            k0, k1 = self._eval_covar_matrix()            
            K0 = self.gamma_down*self.eigvecs0*k0@self.eigvecs0.T
            K1 = self.gamma_up*self.eigvecs1*k1@self.eigvecs1.T
            KK = torch.block_diag(K0, K1)
            K = KK[x1,:][:,x2]

        if diag:
            return K.diag()
        else:
            return K 
        
