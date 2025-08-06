import torch
import torch.nn as nn

class MyFirstModel(nn.Module):
    def __init__(self):
        super(MyFirstModel, self).__init__()

        #defining the layers
        self.layer1=nn.Linear(10,5)
        self.layer2=nn.Linear(5,1)

    def forward(self,x):
        
        x= torch.relu(self.layer1(x))
        x=self.layer2(x)
        return x
    
model=MyFirstModel()
x= torch.randn(2,10)
output=model(x)
print("Input",x)
print ("Output",output) 
