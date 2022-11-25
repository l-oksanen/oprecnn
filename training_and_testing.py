
import numpy as np
import torch
import time
# import datetime

import opnet
from simple_inversion_data import generate_data, save_data, load_data


def training_and_testing(model: opnet.OperatorNet, loss_fn: torch.nn.MSELoss, lr, PATH, train_data_path, test_data_path):
    losses = [] #to save all the avg losses

    # Load the training data
    train_loader = torch.utils.data.DataLoader(
    load_data(train_data_path),
    batch_size=64)

    # dataiter1 = iter(train_loader)
    # a, b = dataiter1.next()
    # print("matriisin ominaisarvot ovat")
    # tulostetaan kääntämätön matriisi ominaisarvotarkastelua varten
    # print(a[:2]) #[:2] tarkoittaa, että listan alusta tulostetaan 2 ekaa alkiota, eli kaksi matriisia

    # Learning rate parameter is from the quickstart guide 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print ("lr= ", lr)
    print("path= ",PATH)

    #otetaan aikaa kauan koko hommaan menee aikaa
    #start_time=time.perf_counter()

    # we want the code to calculate the average loss until the point where
    # the average loss is under certain given point, "limit point", that is small enough
    limit_point = 9e-3
    avg_loss_now = 1 # average loss at this round. Gave some random big number for starters
    loop_counter = 0

    for x in range(25): #for-loop to find the best learning rate
    #while (avg_loss_now)>limit_point: #while-loop to compare relu and no-relu
        loop_counter+=1
        # measure the time consumed in epoch
        #epoch_start = time.perf_counter()
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

        #epoch_end = time.perf_counter()
        # print(f"epochiin kulunut aika: {epoch_end - epoch_start:0.4}")
        # str(datetime.timedelta(seconds=666))

        # measure time consumed in testing
        #testing_start = time.perf_counter()

        torch.save(model.state_dict(), PATH)
        # Load trained variables
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
        avg_loss_now = test_loss

        losses.append(test_loss) #adds a avg loss to the list of losses
        print(f"{(x+1)*2} {test_loss:>8f}") #prints how many epochs has gone + avg loss
        #print(f"{loop_counter*2} {test_loss:>8f}") #prints how many epochs has gone + avg loss

        #testing_end = time.perf_counter()
        # print(f"testaamiseen kulunut aika: {testing_end - testing_start:0.4} \n")
    #end_time=time.perf_counter()
    # print(f"koko hommaan meni aikaa: {end_time - start_time:0.4}")
    return losses





