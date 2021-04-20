import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

def trainer(net, train_loader, valid_loader, optimizer, device, validation=True):
    print("Start Training")


    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(train_loader, 0):
            
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
                  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            y_pred = np.argmax(outputs.detach().cpu().numpy(), 1)
            y_true = labels.detach().cpu().numpy()
            running_accuracy += accuracy_score(y_pred, y_true)
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f accuracy: %.3f' %(epoch + 1, i + 1,
                                                              running_loss / 20,
                                                              running_accuracy / 20))

                running_loss = 0.0
                running_accuracy = 0.0

        if (validation):
                
            with torch.no_grad():
              
              net.eval()
              
              valid_loss = 0
              valid_acc = 0
              for i, data in enumerate(valid_loader, 0):
                
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = net(inputs)
                
                
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                v_y_pred = np.argmax(outputs.detach().cpu().numpy(), 1)
                v_y_true = labels.detach().cpu().numpy()

                valid_acc += accuracy_score(v_y_true, v_y_pred)


              
              print("Validation loss: " + str(valid_loss/i))
              print("Validation accuracy: " + str(valid_acc/i))


    print('Finished Training')
