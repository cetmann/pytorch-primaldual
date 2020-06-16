"""
Simple implementation of the learned primal-dual approach by 
Adler & Öktem (2017), https://arxiv.org/abs/1707.06474
"""

import torch
import torch.nn as nn
import odl.contrib.torch as odl_torch 


class ConcatenateLayer(nn.Module):
    def __init__(self):
        super(ConcatenateLayer, self).__init__()
    
    def forward(self, *x):
        return torch.cat(list(x), dim=1)

class SplitLayer(nn.Module):
    def __init__(self, split_sizes):
        super(SplitLayer, self).__init__()
        self.split_sizes = split_sizes
    
    def forward(self, x):
        current_pos = 0
        chunks = []
        for l in self.split_sizes:
            chunks.append(
                x[:,current_pos : current_pos+l]
            )
            current_pos = l
        return tuple(chunks)

            
class DualNet(nn.Module):
    def __init__(self, n_dual):
        super(DualNet, self).__init__()
        
        self.n_dual = n_dual
        self.n_channels = n_dual + 2
        
        self.input_concat_layer = ConcatenateLayer()
        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, self.n_dual, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)
        
    def forward(self, h, Op_f, g):
        x = self.input_concat_layer(h, Op_f, g)
        x = h + self.block(x)
        return x
    
class PrimalNet(nn.Module):
    def __init__(self, n_primal):
        super(PrimalNet, self).__init__()
        
        self.n_primal = n_primal
        self.n_channels = n_primal + 1
        
        self.input_concat_layer = ConcatenateLayer()
        layers = [
            nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, self.n_primal, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*layers)
        
    def forward(self, f, OpAdj_h):
        x = self.input_concat_layer(f, OpAdj_h)
        x = f + self.block(x)
        return x
        
    

    
    

class LearnedPrimalDual(nn.Module):
    """
    Simple implementation of the learned primal-dual approach by 
    Adler & Öktem (2017), https://arxiv.org/abs/1707.06474
    """
    def __init__(self,
                forward_op,
                primal_architecture = PrimalNet,
                dual_architecture = DualNet,
                n_iter = 10,
                n_primal = 5,
                n_dual = 5):
        
        super(LearnedPrimalDual, self).__init__()
        
        self.forward_op = forward_op
        self.primal_architecture = primal_architecture
        self.dual_architecture = dual_architecture
        self.n_iter = n_iter
        self.n_primal = n_primal
        self.n_dual = n_dual
        
        self.primal_shape = (n_primal,) + forward_op.domain.shape
        self.dual_shape = (n_dual,) + forward_op.range.shape
        
        self.primal_op_layer = odl_torch.OperatorModule(forward_op)
        self.dual_op_layer = odl_torch.OperatorModule(forward_op.adjoint)
        
        self.primal_nets = nn.ModuleList()
        self.dual_nets = nn.ModuleList()
        
        self.concatenate_layer = ConcatenateLayer()
        self.primal_split_layer = SplitLayer([n_primal, n_dual, 1])
        self.dual_split_layer = SplitLayer([n_primal, n_dual])
        
        for i in range(n_iter):
            self.primal_nets.append(
                primal_architecture(n_primal)
            )
            self.dual_nets.append(
                dual_architecture(n_dual)
            )
        
    def forward(self, g, intermediate_values = False):
        
        h = torch.zeros(g.shape[0:1] + (self.dual_shape), device=g.device)
        f = torch.zeros(g.shape[0:1] + (self.primal_shape), device=g.device)
        
        if intermediate_values:
            h_values = []
            f_values = []
            
        for i in range(self.n_iter):
            ## Dual
            # Apply forward operator to f^(2)
            f_2 = f[:,1:2]
            if intermediate_values:
                f_values.append(f)
            Op_f = self.primal_op_layer(f_2)
            # Apply dual network
            h = self.dual_nets[i](h, Op_f, g)
            
            ## Primal
            # Apply adjoint operator to h^(1)
            h_1 = h[:,0:1]
            if intermediate_values:
                h_values.append(h)
            OpAdj_h = self.dual_op_layer(h_1)
            # Apply primal network
            f = self.primal_nets[i](f, OpAdj_h)
        
        if intermediate_values:
            return f[:,0:1], f_values, h_values
        return f[:,0:1]
        