#Predict 1 if the input is greater than or equal to 0.7, else predict 0.
import torch
import torch.nn as nn
import torch.optim as optim

class classifierModel(nn.Module):
    def __init__(self):
        super(classifierModel,self).__init__()
        self.layer=nn.Linear(1,1)
    def forward(self,x):
        y=self.layer(x)
        return torch.sigmoid(y)
    
#trainig data
x_train=torch.tensor([[0.1],[0.2],[0.01],[0.6],[0.7],[0.8],[0.89]],dtype=torch.float)
y_train=torch.tensor([[0],[0],[0],[0],[1],[1],[1]],dtype=torch.float)


#creating model
model=classifierModel()

#binary cross entropy for classification
error=nn.BCELoss()

#optimizer
optimizer=optim.SGD(model.parameters(),lr=0.1)

#training loop
epoch=1000
for i in range(epoch):
    y_predicted=model(x_train)
    loss=error(y_predicted,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#testing the model
testdata=torch.tensor([[0.7],[0.6]])
predicted=model(testdata)
print(f"{predicted}")