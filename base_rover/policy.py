import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x