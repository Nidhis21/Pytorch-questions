import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris=load_iris()
X=iris.data #Gives you the features of the flower
Y=iris.target #Gives you the label of the flower

#Standardising data
scaler = StandardScaler()
X = scaler.fit_transform(X)  #centered around 0 and standard deviation of 1

#training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#converting arrays to tensors
X_train=torch.tensor(X_train,dtype=torch.float32)   
Y_train=torch.tensor(Y_train,dtype=torch.long)
X_test=torch.tensor(X_test,dtype=torch.float32)
Y_test=torch.tensor(Y_test,dtype=torch.long)

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,3)
        )
    def forward(self,x):
        return self.network(x)
    
model=IrisNet()

#Multiple classes error claculation
error=nn.CrossEntropyLoss()

#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training data
epoch = 1000
for i in range(epoch):
    y_pred=model(X_train)
    loss=error(y_pred,Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#testing model
with torch.no_grad():
    y_pred=model(X_test)
    _,predicted=torch.max(y_pred,1)
    correctpred=(predicted==Y_test)
    num_correct=correctpred.sum().item()
    total=Y_test.size(0)
    accuracy=num_correct/total
    print(f"Accuracy: {accuracy}")

class_name=['Setosa', 'Versicolor', 'Virginica']
new_data= scaler.transform([[5.1, 3.5, 1.4, 0.2]])
new_tensors=torch.tensor(new_data,dtype=torch.float32)

with torch.no_grad():
    predicted=model(new_tensors)
    probabilities = F.softmax(predicted, dim=1)
    value=torch.argmax(probabilities,1).item()
    print(f"{class_name[value]}")



