
import numpy as np
import torch

import opnet
from volume_inversion_data import generate_data, save_data, load_data

def wave_training_and_testing(model: opnet.OperatorNet, loss_fn: torch.nn.MSELoss, lr, PATH, train_data_path, test_data_path):
    # Load the training data
    train_loader = torch.utils.data.DataLoader(
    load_data(train_data_path),
    batch_size=64)

    # Learning rate parameter is from the quickstart guide
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # we want the code to calculate the average loss until the point where
    # the average loss is under sertain given point "limitpoint"
    limit_point = 1e-5
    avg_loss_pre = 10 # average loss at the previous round. Gave some random big number for starters
    avg_loss_now = 9 # average loss at this round. Gave some random big number for starters
    loop_counter = 0
    while (avg_loss_pre - avg_loss_now)>limit_point:
        loop_counter+=1
        for epoch in range(20):
            # print(f"Epoch {epoch+1}\n-------------------------------")
            for batch, (X, y) in enumerate(train_loader):
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Print statistics
                if batch % 10 == 0:
                    n, N = (batch + 1) * len(X), len(train_loader.dataset)
                    # print(f"loss: {loss.item():>7f}  [{n:>5d}/{N:>5d}]")

        torch.save(model.state_dict(), PATH)

        ## Load trained variables
        model.load_state_dict(torch.load(PATH))

        # Load the testing data
        test_loader = torch.utils.data.DataLoader(
            load_data(test_data_path), 
            batch_size=64)

        dataiter = iter(test_loader)
        X, y = dataiter.next()
        with torch.no_grad():
            pred = model(X)
        # print("True: ")
        # print(y[:2])
        # print("Prediction: ")
        # print(pred[:2])

        num_batches = len(test_loader)
        test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        avg_loss_pre = avg_loss_now
        avg_loss_now = test_loss
        print(f"{loop_counter*20} {test_loss:>8f}") #antaa kierrosen ja avglossin
