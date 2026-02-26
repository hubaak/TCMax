import torch
import torch.nn as nn
import math

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output
    
    def forward_a(self, a):
        return self.fc_x(a)
    def forward_v(self, v):
         return self.fc_y(v)
     


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100, args = None):
        super(ConcatFusion, self).__init__()
        self.args =args
        self.input_dim = input_dim
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output
    
    def forward_a(self, a):
        return (torch.mm(a, torch.transpose(self.fc_out.weight[:, :self.input_dim//2], 0, 1))
                    + self.fc_out.bias/2)
    def forward_v(self, v):
        return (torch.mm(v, torch.transpose(self.fc_out.weight[:, self.input_dim//2:], 0, 1))
                    + self.fc_out.bias/2)
        
class ConcatFusion_with_modality_specific_head(nn.Module):
    def __init__(self, input_dim=512, output_dim=100, args = None):
        super(ConcatFusion_with_modality_specific_head, self).__init__()
        self.args =args
        self.fc_out = nn.Linear(input_dim*2, output_dim)
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output
    
    def forward_a(self, a):
        return self.fc_x(a)
    def forward_v(self, v):
         return self.fc_y(v)


class Share_Head(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(Share_Head, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
    
    def forward(self, x, y, alpha=0.5):
        output_x = self.fc_out(x)
        output_y = self.fc_out(y)
        return x, y, alpha * output_x + (1-alpha) * output_y
    
    def forward_a(self, a):
        return self.fc_out(a)
    def forward_v(self, v):
         return self.fc_out(v)


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """
    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()
        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)
        self.x_film = x_film

    def forward(self, x, y):
        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)
        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output
    
    def forward_a(self, a):
        if self.x_film:
            film = a
            to_be_film = torch.zeros_like(a)
        else:
            film = torch.zeros_like(a)
            to_be_film = a
        gamma, beta = torch.split(self.fc(film), self.dim, 1)
        output = gamma * to_be_film + beta
        output = self.fc_out(output)
        return output
    
    def forward_v(self, v):
        if self.x_film:
            film = torch.zeros_like(v)
            to_be_film = v
        else:
            film = v
            to_be_film = torch.zeros_like(v)
        gamma, beta = torch.split(self.fc(film), self.dim, 1)
        output = gamma * to_be_film + beta
        output = self.fc_out(output)
        return output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return x, y, output
    
    def forward_a(self, a):
        out_x = self.fc_x(a)
        out_y = self.fc_y(torch.zeros_like(a))
        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))
        return output
    
    def forward_v(self, v):
        out_x = self.fc_x(torch.zeros_like(v))
        out_y = self.fc_y(v)
        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))
        return output
    

class BiGateFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100):
        super(BiGateFusion, self).__init__()

        self.fc_xgate_x = nn.Linear(input_dim, dim)
        self.fc_xgate_y = nn.Linear(input_dim, dim)
        self.fc_xgate_out = nn.Linear(dim, output_dim)
        
        self.fc_ygate_x = nn.Linear(input_dim, dim)
        self.fc_ygate_y = nn.Linear(input_dim, dim)
        self.fc_ygate_out = nn.Linear(dim, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x_xgate = self.fc_xgate_x(x)
        out_y_xgate = self.fc_xgate_y(y)
        gate_xgate = self.sigmoid(out_x_xgate)
        output_xgate = self.fc_xgate_out(torch.mul(gate_xgate, out_y_xgate))
        
        out_x_ygate = self.fc_ygate_x(x)
        out_y_ygate = self.fc_ygate_y(y)
        gate_ygate = self.sigmoid(out_y_ygate)
        output_ygate = self.fc_ygate_out(torch.mul(gate_ygate, out_x_ygate))
        
        output = output_xgate + output_ygate

        return x, y, output
    
    def forward_a(self, a):
        out_x = self.fc_ygate_x(a)
        out_y = self.fc_ygate_y(torch.zeros_like(a))
        gate = self.sigmoid(out_y)
        output = self.fc_ygate_out(torch.mul(out_x, gate))
        return output
    
    def forward_v(self, v):
        out_x = self.fc_xgate_x(torch.zeros_like(v))
        out_y = self.fc_xgate_y(v)
        gate = self.sigmoid(out_x)
        output = self.fc_xgate_out(torch.mul(gate, out_y))
        return output



class CLIP_Predict_Head(nn.Module):
    def __init__(self, input_dim=512, output_dim=100, temp=0.07):
        super(CLIP_Predict_Head, self).__init__()
        self.class_num = output_dim
        self.category_features_x = torch.randn(self.class_num, input_dim).float()
        self.category_features_x = self.category_features_x / self.category_features_x.norm(dim=-1, keepdim=True)
        self.category_features_x = nn.Parameter(self.category_features_x)
        self.category_features_y = torch.randn(self.class_num, input_dim).float()
        self.category_features_y = self.category_features_y / self.category_features_y.norm(dim=-1, keepdim=True)
        self.category_features_y = nn.Parameter(self.category_features_y)
        self.temp = temp
    
    def forward(self, a, v):
        return a, v, self.forward_a(a) + self.forward_v(v)
    
    def forward_a(self, a):
        B = a.shape[0]
        C = self.class_num
        a = a / a.norm(dim=-1, keepdim=True) # [B, D]
        category_features = self.category_features_x / self.category_features_x.norm(dim=-1, keepdim=True)
        output = a @ category_features.t() / self.temp
        return output
    
    def forward_v(self, v):
        B = v.shape[0]
        C = self.class_num
        v = v / v.norm(dim=-1, keepdim=True) # [B, D]
        category_features = self.category_features_y / self.category_features_y.norm(dim=-1, keepdim=True)
        output = v @ category_features.t() / self.temp
        return output
        

