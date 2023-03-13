import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

import torch.jit

traced_script_module = torch.jit.trace(model, torch.randn(1, 2))
traced_script_module.save("model.pt")
