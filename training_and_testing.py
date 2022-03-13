
import numpy as np
import torch
import time
# import datetime

import opnet
from simple_inversion_data import generate_data, save_data, load_data

PATH = './simple_inversion_net.pth'

def training_and_testing(model: opnet.OperatorNet, loss_fn: torch.nn.MSELoss, lr):
    # Load the training data
    train_loader = torch.utils.data.DataLoader(
    load_data("simple_inversion_train_data.npz"), 
    batch_size=64)

    # Learning rate parameter is from the quickstart guide 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print ("lr= ", lr)

    #otetaan aikaa kauan koko hommaan menee aikaa
    start_time=time.perf_counter()

    for x in range(20):
        print("kierros ", x+1)
        #mitataan looppiin kulunut aika
        epoch_start = time.perf_counter()
        for epoch in range(30): 
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
                # if batch % 100 == 0:
                #     n, N = (batch + 1) * len(X), len(train_loader.dataset)
                #     print(f"loss: {loss.item():>7f}  [{n:>5d}/{N:>5d}]")

        epoch_end = time.perf_counter()
        print(f"epochiin kulunut aika: {epoch_end - epoch_start:0.4}")
        # str(datetime.timedelta(seconds=666))

        #mitataan testaamiseen kulunut aika
        testing_start = time.perf_counter()

        torch.save(model.state_dict(), PATH)

        # Load trained variables
        model.load_state_dict(torch.load(PATH))

        # Load the testing data
        test_loader = torch.utils.data.DataLoader(
        load_data("simple_inversion_test_data.npz"), 
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
        print(f"Avg loss: {test_loss:>8f}")

        testing_end = time.perf_counter()
        print(f"testaamiseen kulunut aika: {testing_end - testing_start:0.4} \n")

    end_time=time.perf_counter()
    print(f"koko hommaan meni aikaa: {end_time - start_time:0.4}")





