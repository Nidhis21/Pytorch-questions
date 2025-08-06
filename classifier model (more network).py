import torch
import torch.nn as nn
import torch.optim as optim

#creating a model
class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.network(x)
    
#Training Data
x_train=torch.tensor([[0.1],[0.2],[0.3],[0.4],[0.45],[0.55],[0.65],[0.9]])
y_train=torch.tensor([[0],[0],[0],[0],[0.5],[0.5],[1],[1]])

#model
model=ClassifierModel()

#error
error=nn.MSELoss() #MSE to support neutal value of 0.5

#optimizer
optimizer=optim.Adam(model.parameters(),lr=0.01)

#training loop
epoch=1000
for i in range(epoch):
    y_predicted=model(x_train)
    loss=error(y_predicted,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#testing model
testdata=torch.tensor([[0.7],[0.5],[0.4],[0.35]])
predicted=model(testdata)
print(f"{predicted}")

