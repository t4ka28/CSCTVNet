import util
import math
import torch
import IPython

import torch.nn as nn
from collections import namedtuple

CSCTVParams = namedtuple('CSC_TV_Box_epsParams', ['deg', 'n_channels','n_filters','kernel_size','K'])
CSCDVTVParams = namedtuple('CSC_DVTV_epsParams', ['deg', 'n_channels','n_filters','kernel_size','K','w'])

class CSCTV(nn.Module):

    def __init__(self, params: CSCTVParams):
        super(CSCTV, self).__init__()

        # IPython.embed()
        # exit(0)
        self.B = nn.Parameter(torch.normal(mean=0, std=0.01, size=(params.n_filters, params.n_channels, params.kernel_size, params.kernel_size)))
        op_norm_B = util.calc_max_singular_P(self.B[:,0])
        # λ
        lam1, lam2 = 1e-2, 1e-2
        self.lam1 = nn.Parameter(torch.ones(params.K)*lam1)
        self.lam2 = nn.Parameter(torch.ones(params.K)*lam2)
                        
        # γ
        beta = (1+op_norm_B)**2
        gam1 = 2/beta*0.9
        gam2 = 2/beta*0.9
        gam3 = 1/8*(1/gam1 - beta/2)
        
        self.gam1 = nn.Parameter(torch.ones(params.K)*gam1)
        self.gam2 = nn.Parameter(torch.ones(params.K)*gam2)
        self.gam3 = nn.Parameter(torch.ones(params.K)*gam3)
        
        self.params = params

    def forward(self, z, phi, alpha=0.2):
        '''
        z : mb x channel x N x M
        a : mb x P x channel x N x M
        B : P x channel x k x k
        '''
        # IPython.embed()
        # exit(0)
        eps = 1e-8
        mb, channels, M, N = z.shape 
        P = self.B.shape[0]
        x_bef = z
        a_bef = torch.normal(mean=0, std=0.01, size=(mb, P, channels, M, N))
        y1_bef = torch.zeros(mb, 2, channels, M, N)
        y2_bef = torch.zeros(mb, channels, M, N)
        # ε
        epsilon = alpha*math.sqrt(channels*M*N)
        
        # for k in range(self.params.K):
        #     a = a_bef - self.gam[k]*util.convT_CSC(phi(phi(util.conv_CSC(a_bef, self.B) - z),T=True), self.B)
        #     a_aft = util.ProxL1(a, self.lam[k]*self.gam[k])
        #     a_bef = a_aft

        # return util.conv_CSC(a_aft, self.B)

        for k in range(self.params.K):
            x = x_bef - self.gam1[k]*((x_bef - util.conv_CSC(a_bef, self.B)) + util.Dt(y1_bef) + phi(y2_bef, T=True))
            x_aft = util.ProxBoxConstraint(x)

            a = a_bef + self.gam2[k]*util.convT_CSC(x_bef - util.conv_CSC(a_bef, self.B), self.B)
            a_aft = util.ProxL1(a, self.gam2[k]*self.lam1[k])

            y1 = y1_bef + self.gam3[k]*util.D(2*x_aft - x_bef)
            y1_aft = y1 - self.gam3[k]*util.ProxL12(y1/(self.gam3[k]+eps), self.lam2[k]/(self.gam3[k]+eps))

            y2 = y2_bef + self.gam3[k]*phi(2*x_aft - x_bef)
            y2_aft = y2 - self.gam3[k]*util.ProjL2ball(y2/(self.gam3[k]+eps), z, epsilon)

            x_bef = x_aft   
            a_bef = a_aft
            y1_bef = y1_aft
            y2_bef = y2_aft

        return x_aft

class CSCDVTV(nn.Module):

    def __init__(self, params: CSCDVTVParams):
        super(CSCDVTV, self).__init__()

        self.B = nn.Parameter(torch.normal(mean=0, std=0.01, size=(params.n_filters, params.n_channels, params.kernel_size, params.kernel_size)))
        op_norm_B = 0
        for c in range(params.n_channels):
            op_norm_B += util.calc_max_singular_P(self.B[:,c])**2
        op_norm_B = torch.sqrt(op_norm_B)
        # λ
        lam1, lam2 = 5e-4, 5e-4
        self.lam1 = nn.Parameter(torch.ones(params.K)*lam1)
        self.lam2 = nn.Parameter(torch.ones(params.K)*lam2)
                        
        # γ
        beta = 1 + 2*op_norm_B + op_norm_B**2
        gam1 = 2/beta*0.9
        gam2 = 2/beta*0.9
        gam3 = 1/(1+4*2+params.w*4*2)*(1/gam1 - beta/2)
        self.gam1 = nn.Parameter(torch.ones(params.K)*gam1)
        self.gam2 = nn.Parameter(torch.ones(params.K)*gam2)
        self.gam3 = nn.Parameter(torch.ones(params.K)*gam3)

        self.params = params

    def forward(self, z, phi, alpha=0.02):
        '''
        z : mb x channel x N x M
        a : mb x P x channel x N x M
        B : P x channel x k x k
        '''
        eps = 1e-8
        mb, channels, M, N = z.shape 
        P = self.B.shape[0]
        x_bef = z
        a_bef = torch.normal(mean=0, std=0.01, size=(mb, P, channels, M, N))
        y1_bef = torch.zeros(mb, 2, channels, M, N)
        y2_bef = torch.zeros(mb, channels, M, N)
        # ε
        epsilon = alpha*math.sqrt(channels*M*N)
        
        if self.params.deg == "inpainting":

            for k in range(self.params.K):
                x = x_bef - self.gam1[k]*(x_bef - util.conv_CSC(a_bef, self.B) + util.Ct(util.Dt(y1_bef)) + phi(y2_bef, T=True))
                x_aft = util.ProxBoxConstraint(x)

                a = a_bef + self.gam2[k]*util.convT_CSC(x_bef - util.conv_CSC(a_bef, self.B), self.B)
                a_aft = util.ProxL1(a, self.gam2[k]*self.lam1[k])

                y1 = y1_bef + self.gam3[k]*util.D(util.C(2*x_aft - x_bef))
                y1_aft = y1 - self.gam3[k]*util.ProxDVTVnorm(y1/(self.gam3[k]+eps), self.lam2[k]/(self.gam3[k]+eps), self.params.w)

                y2 = y2_bef + self.gam3[k]*phi(2*x_aft - x_bef)
                y2_aft = y2 - self.gam3[k]*util.ProjL2ball(y2/(self.gam3[k]+eps), z, epsilon)

                x_bef = x_aft   
                a_bef = a_aft
                y1_bef = y1_aft
                y2_bef = y2_aft

        return x_aft