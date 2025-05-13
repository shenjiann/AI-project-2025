import torch
import torch.nn as nn

class FNN(nn.Module):
    """
    A fully connected neural network for binary classification.
    """
    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(24, 64) # input layer: 25 features -> 64 neurons
        self.fc2 = nn.Linear(64, 32) # hidden layer1: 64 neurons -> 32 neurons
        self.fc3 = nn.Linear(32, 8) # hidden layer2: 32 neurons -> 8 neurons
        self.fc4 = nn.Linear(8, 1) # hidden layer3: 8 neurons -> output probability
        
    def forward(self, x):
        x = torch.relu(self.fc1(x)) # relu activation
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x)) # sigmoid activation
        return x

def train(model, train_loader, criterion, optimizer, num_epochs):
    """
    functions used to train the model
    params:
        model: the neural network to be trained;
        train_loader: a DataLoader object, trainging dataset;
        criterion: loss function;
        optimizer: optimiser that updates parameters after backpropagation;
        num_epochs: number of epochs in optimization;
    return:
        None
    """
    model.train() # set the model to training mode
    for epoch in range(num_epochs): # loop over epochs
        for batch_idx, (data, target) in enumerate(train_loader): # loop over batches
            optimizer.zero_grad() # clear previous gradients
            output = model(data) # do forward propagation
            loss = criterion(output.squeeze(1), target) # calculate loss
            loss.backward() # do backpropagation
            optimizer.step() # update parameters

            if (epoch + 1) % 50 == 0 and batch_idx == len(train_loader)-2: # print second last batch's loss every 50 epochs
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx+1} Loss: {loss.item():.4f}")

def test(model, test_loader, criterion):
    """
    Run the trained model on the test set.
    params:
        model: the network after trained;
        test_loader: DataLoader, test dataset;
        criterion: loss function;
    return:
        test_loss: float, loss on the test data
        pred_probs: tensor, predicted test probabilities;
    """
    model.eval() # set the model to evaluation mode
    acc_loss = 0
    pred_probs = [] # a list used to accumulated each batch's predicted probabilities
    with torch.no_grad(): # not calculate gradient
        for data, target in test_loader:
            output = model(data) # forward propagation using test data and return output probabilities
            loss = criterion(output.squeeze(1), target).item() # accumulate loss
            acc_loss += loss * data.size(0)
            pred_probs.append(output) # append predicted probabilityes

    test_loss = acc_loss / len(test_loader.dataset)
    
    return test_loss, torch.cat(pred_probs)