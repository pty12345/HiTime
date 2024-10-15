import torch, copy
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=40):
        super(InceptionModule, self).__init__()
        ks = [10, 20, 40]
        # ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        self.convs = nn.ModuleList([nn.Conv1d(ni, nf, k, bias=False, padding='same') for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(ni, nf, 1, bias=False, padding='same')])
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = torch.cat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)], dim=1)
        # print(x.shape); exit(0)
        return self.act(self.bn(x))
    
class InceptionBlock(nn.Module):
    def __init__(self, ni, nf=32, residual=True, depth=3, **kwargs):
        super(InceptionBlock, self).__init__()
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList([]), nn.ModuleList([])
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * 4, nf, **kwargs))
            if self.residual and d % 3 == 2: 
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out else nn.Conv1d(n_in, n_out, 1, padding='same'))
        self.act = nn.ReLU()
        
    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2: res = x = self.act(x + (self.shortcut[d//3](res)))
        return x
    
class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__()
        self.c_in = configs.enc_in
        self.c_out = configs.num_class
        
        nf = configs.d_model
        depth = configs.e_layers
        
        
        if configs.use_patch:
            patch_size = configs.patch_size
            patch_stride = int(0.5 * configs.patch_size)
            self.patch_layer = nn.Sequential(
                nn.Conv1d(self.c_in, nf, patch_size, patch_stride),
                nn.Dropout(configs.dropout)
            )
            self.inceptionblock = InceptionBlock(nf, nf, depth=depth, **kwargs)
        else:
            self.patch_layer = nn.Identity()
            self.inceptionblock = InceptionBlock(self.c_in, nf, depth=depth, **kwargs)
            
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nf * 4, self.c_out)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        x = x.permute(0,2,1) if x.shape[-1] == self.c_in else x
        x = self.patch_layer(x)
        x = self.inceptionblock(x)
        x = self.gap(x).permute(0,2,1)
        
        x = self.fc(x).squeeze(1)
        return x