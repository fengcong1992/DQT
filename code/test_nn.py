import torch
import torch.nn as nn

class output_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12,24)
        self.l2 = nn.Linear(24, 10)
    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        return output
        
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layers = output_layer()
        self.linears = nn.ModuleList([self.output_layers for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints 
        out = []
        for i, l in enumerate(self.linears):
            if i == 0:
                out = l(x) 
            else:
                out = torch.cat([out,l(x)])
        return out
    
model = MyModule()
x = torch.randn([20,12])
out = model(x)
print(model)
print(out.shape)