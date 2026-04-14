import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFusionLayer(nn.Module):
    def __init__(self, in_dim, prev_bits, out_bits):
        super().__init__()
        emb_weights = torch.ones(in_dim) * 0.5
        hash_weights = torch.ones(prev_bits) * 1.5
        full_weights = torch.cat([emb_weights, hash_weights], dim=0)
        self.weights = nn.Parameter(full_weights)
        
        self.fc = nn.Sequential(
            nn.Linear(in_dim + prev_bits, out_bits),
            nn.Tanh()
        )
    
    def forward(self, x, prev_hash):
        concat = torch.cat([x, prev_hash], dim=1)
        weighted = concat * self.weights
        return self.fc(weighted)

class AttentionGatedCrossFusion(nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        
        self.cross_attn = nn.Sequential(
            nn.Linear(bits, bits),
            nn.ReLU(inplace=True)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(3 * bits, bits),  # 输入: [a, b, c]
            nn.Sigmoid()
        )
        
        self.enhance = nn.Sequential(
            nn.LayerNorm(bits),
            nn.Linear(bits, bits),
            nn.Tanh()
        )

    def forward(self, a, b):
        element_wise = a * b  
        c = self.cross_attn(element_wise)
        
        combined = torch.cat([a, b, c], dim=1)
        
        g = self.gate(combined)
        
        fused = g * a + (1 - g) * b
        
        return fused

class MultiScaleHashFusionModule(nn.Module):
    def __init__(self, in_dim: int = 512, outputDim: int = 64):
        super().__init__()
        assert outputDim in [8, 16, 32, 64, 128], "outputDim 必须为 8,16,32,64 或 128"
        self.in_dim = in_dim
        self.outputDim = outputDim
        
        self.bu_layers = self._create_bu_path()
        
        self.td_layers = self._create_td_path()
        
        self.fuse_8 = AttentionGatedCrossFusion(8)
        self.fuse_16 = AttentionGatedCrossFusion(16)
        self.fuse_32 = AttentionGatedCrossFusion(32)
        self.fuse_64 = AttentionGatedCrossFusion(64)
        self.fuse_128 = AttentionGatedCrossFusion(128)

    def _create_bu_path(self):
        layers = nn.ModuleDict()
        
        layers['8'] = nn.Sequential(
            nn.Linear(self.in_dim, 8),
            nn.Tanh()
        )
        
        layers['16'] = WeightedFusionLayer(self.in_dim, 8, 16)
        
        layers['32'] = WeightedFusionLayer(self.in_dim, 16, 32)
        
        layers['64'] = WeightedFusionLayer(self.in_dim, 32, 64)
        
        layers['128'] = WeightedFusionLayer(self.in_dim, 64, 128)
        
        return layers

    def _create_td_path(self):
        layers = nn.ModuleDict()
        
        layers['128'] = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.Tanh()
        )
        
        layers['64'] = WeightedFusionLayer(self.in_dim, 128, 64)
        
        layers['32'] = WeightedFusionLayer(self.in_dim, 64, 32)
        
        layers['16'] = WeightedFusionLayer(self.in_dim, 32, 16)
        
        layers['8'] = WeightedFusionLayer(self.in_dim, 16, 8)
        
        return layers

    def forward(self, x: torch.Tensor, return_all=False):
        
        bu_8 = self.bu_layers['8'](x)
        bu_16 = self.bu_layers['16'](x, bu_8)
        bu_32 = self.bu_layers['32'](x, bu_16)
        bu_64 = self.bu_layers['64'](x, bu_32)
        bu_128 = self.bu_layers['128'](x, bu_64)
        
        td_128 = self.td_layers['128'](x)
        td_64 = self.td_layers['64'](x, td_128)
        td_32 = self.td_layers['32'](x, td_64)
        td_16 = self.td_layers['16'](x, td_32)
        td_8 = self.td_layers['8'](x, td_16)
        
        fused_8 = self.fuse_8(bu_8, td_8)
        fused_16 = self.fuse_16(bu_16, td_16)
        fused_32 = self.fuse_32(bu_32, td_32)
        fused_64 = self.fuse_64(bu_64, td_64)
        fused_128 = self.fuse_128(bu_128, td_128)
        
        if self.outputDim == 8:
            final_hash = fused_8
        elif self.outputDim == 16:
            final_hash = fused_16
        elif self.outputDim == 32:
            final_hash = fused_32
        elif self.outputDim == 64:
            final_hash = fused_64
        else:  # 128
            final_hash = fused_128
        
        scale_hashes = {
            '8': fused_8,
            '16': fused_16,
            '32': fused_32,
            '64': fused_64,
            '128': fused_128
        }
        
        if return_all:
            return final_hash, scale_hashes
        return final_hash