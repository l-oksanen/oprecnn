
import numpy as np
import torch
import time
# import datetime

import opnet
from simple_inversion_data import generate_data, save_data, load_data

PATH = './simple_inversion_net4.pth'
# PATH = './simple_inversion_netPlusMinusReLU.pth'
# PATH = './simple_inversion_netNEGA.pth'
# PATH = './simple_inversion_netReLU.pth'

def training_and_testing(model: opnet.OperatorNet, loss_fn: torch.nn.MSELoss, lr):
    # Load the training data
    train_loader = torch.utils.data.DataLoader(
    load_data("simple_inversion_train_dataPOSNEG.npz"), 
    batch_size=64)

    # dataiter1 = iter(train_loader)
    # a, b = dataiter1.next()
    # print("matriisin ominaisarvot ovat")
    # tulostetaan kääntämätön matriisi ominaisarvotarkastelua varten
    # print(a[:2]) #[:2] tarkoittaa, että listan alusta tulostetaan 2 ekaa alkiota, eli kaksi matriisia

    # Learning rate parameter is from the quickstart guide 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print ("lr= ", lr)

    #otetaan aikaa kauan koko hommaan menee aikaa
    start_time=time.perf_counter()

    for x in range(2):
        print("kierros ", x+1)
        # measure the time consumed in epoch
        epoch_start = time.perf_counter()
        for epoch in range(2): 
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
                #    n, N = (batch + 1) * len(X), len(train_loader.dataset)
                #    print(f"loss: {loss.item():>7f}  [{n:>5d}/{N:>5d}]")

        epoch_end = time.perf_counter()
        # print(f"epochiin kulunut aika: {epoch_end - epoch_start:0.4}")
        # str(datetime.timedelta(seconds=666))

        # measure time consumed in testing
        testing_start = time.perf_counter()

        torch.save(model.state_dict(), PATH)
    
        # Load trained variables
        model.load_state_dict(torch.load(PATH))

        # Load the testing data
        test_loader = torch.utils.data.DataLoader(
        load_data("simple_inversion_test_dataPOSNEG.npz"), 
        batch_size=64)

        dataiter = iter(test_loader)
        X, y = dataiter.next()
        with torch.no_grad():
            pred = model(X)
        print("True: ")
        print(y[:2])
        print("Prediction: ")
        print(pred[:2])

        num_batches = len(test_loader)
        test_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Avg loss: {test_loss:>8f}")

        testing_end = time.perf_counter()
        # print(f"testaamiseen kulunut aika: {testing_end - testing_start:0.4} \n")

    end_time=time.perf_counter()
    # print(f"koko hommaan meni aikaa: {end_time - start_time:0.4}")





