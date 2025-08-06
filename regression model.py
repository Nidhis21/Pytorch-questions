#simple regression (dependent and independent variables) model
import torch
import torch.nn as nn
import torch.optim as optim

class regressionModel(nn.Module):
    def __init__(self):
        super(regressionModel, self).__init__()
        self.layer1=nn.Linear(1,1)
    
    def forward(self,x):
        return self.layer1(x)

#creating training data
#Data for y=2x+1
x=torch.tensor([[1.0],[2.0],[3.0]],dtype=torch.float)
y=torch.tensor([[3.0],[5.0],[7.0]],dtype=torch.float)

#creating model
model=regressionModel()
#mean squared error for regression
error=nn.MSELoss()

#optimizer
optimizer=optim.SGD(model.parameters(),lr=0.01)

#training loop
epoch =1000
for i in range(epoch):
    valueOfy=model(x)
    loss=error(valueOfy,y)
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()


#testing the model
testdata=torch.tensor([[26.0]])
predicted=model(testdata)
print(f"{predicted.item():.2f}")