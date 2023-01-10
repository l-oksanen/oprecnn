
import numpy as np
import torch

import opnet
from volume_inversion_data import generate_data, save_data, load_data

def wave_training_and_testing(model: opnet.OperatorNet, loss_fn: torch.nn.MSELoss, lr, PATH, train_data_path, test_data_path):
    losses = [] #to save all the avg losses
    
    # Load the training data
    train_loader = torch.utils.data.DataLoader(
    load_data(train_data_path),
    batch_size=64)

    # Learning rate parameter is from the quickstart guide
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print ("lr= ", lr)
    print("path= ",PATH)

    # we want the code to calculate the average loss until the point where
    # the average loss is under certain given point, "limit point", that is small enough
    limit_point = 9e-5
    avg_loss_now = 1 # average loss at this round. Gave some random big number for starters
    loop_counter = 0

    #for x in range(25): #for-loop to find the best learning rate
    while (avg_loss_now)>limit_point:
        loop_counter+=1
        for epoch in range(2):
            for batch, (X, y) in enumerate(train_loader):
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        torch.save(model.state_dict(), PATH)
        model.load_state_dict(torch.load(PATH)) # Load trained variables

        # Load the testing data
        test_loader = torch.utils.data.DataLoader(
            load_data(test_data_path), 
            batch_size=64)

        dataiter = iter(test_loader)
        X, y = dataiter.next()
        with torch.no_grad():
            pred = model(X)

        num_batches = len(test_loader)
        test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        avg_loss_now = test_loss

        losses.append(test_loss) #adds a avg loss to the list of losses
        #print(f"{(x+1)*2} {test_loss:>8f}") #prints how many epochs has gone + avg loss
        print(f"{loop_counter*2} {test_loss:>8f}") #prints how many epochs has gone + avg loss
    
    return losses
