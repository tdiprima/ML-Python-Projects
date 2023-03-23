class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=8)  # takes in 2 features (X), produces 8 features
        self.layer_2 = nn.Linear(in_features=8, out_features=1)  # takes in 8 features, produces 1 feature (y)

    # Define a forward method containing the forward pass computation
    def forward(self, x):
        return self.layer_2(self.layer_1(x))
