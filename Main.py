
from sklearn.utils import resample
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

def load_initial_data():

    def unpickle():
        import pickle
        with open('data_batch_1', 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    a = unpickle()

    data = a[b'data']
    labels =  np.array(a[b'labels'])
    batch_label = np.array(a[b'batch_label'])
    return data,labels


    

def create_model(toggle:bool = False):
    class ConvolutionalNeuralNetwork(nn.Module):
        def __init__(self,print_toggle:bool = False) -> None:
            super().__init__()
            #3 * 32 * 32
            self.conv1 = nn.Conv2d(3,30,kernel_size=3,padding=1) 
            self.conv2 = nn.Conv2d(30,20,kernel_size=3,padding=1)

            self.conv3 = nn.ConvTranspose2d(20,30, kernel_size=2,stride=2)
            self.conv4 = nn.ConvTranspose2d(30,3, kernel_size=2,stride=2)

            self.print = print_toggle

        def forward(self,x):
            if self.print : print(x.shape)
            x = self.conv1(x)
            if self.print : print(x.shape)
            x = F.max_pool2d(x,2)
            x = F.leaky_relu(x)
            if self.print : print(x.shape)
            # 1st block
            x = self.conv2(x)
            if self.print : print(x.shape)
            x = F.max_pool2d(x,2)
            x = F.leaky_relu(x)
            if self.print : print(x.shape)
            # 2nd block

            x = self.conv3(x)
            if self.print : print(x.shape)
            x = F.leaky_relu(x)

            x = self.conv4(x)
            if self.print : print(x.shape)
            
            return x

    net = ConvolutionalNeuralNetwork(toggle)
    lossfun = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=.01)

    return net,lossfun,optimizer



def train_model(net, lossfunction, optimizer, train_loader, device, epochs:int = 20):

    losses = []

    for epochi in range(epochs):
        print(epochi)
        batch_loss = []
        batch_accuracy = []
        i = 0
        for X,y in train_loader:
            # X.shape 32*3*32*32
            #print(X.shape)
            i+=1
            if i%50 == 0 :print(f'epoch {epochi} and batch {i}')

            X = X.to(device)
            y = y.to(device)

            yHat = net(X)
            #print(yHat.shape)
            yHat = yHat.reshape((32,3,32,32))
            #print(yHat.shape)
            loss = lossfunction(yHat,X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

        batch_loss = torch.tensor(batch_loss)
        batch_loss = batch_loss.to('cpu')

        #print(batch_accuracy.device)
        print(torch.mean(batch_loss))
        losses.append (torch.mean(batch_loss))
    
    return losses

def data_processing(data,labels):

    dataT = torch.tensor(data).float()
    labelsT = torch.tensor(labels).long()

    dataT = dataT.reshape((dataT.shape[0],3,32,32))

    train_data,test_data,train_labels,test_labels = train_test_split(dataT,labelsT,test_size=.1)


    train_data = TensorDataset(train_data,train_labels)
    test_data = TensorDataset(test_data,test_labels)
    
    batchsize = 32
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

    return train_loader,test_loader

def plot_lists(*args):
    a = (len(args))
    fig,ax = plt.subplots(a)

    if a == 1:
        ax.plot(*args,'o')
        plt.show()
        return

    i = 0
    for list in args:
        ax[i].plot(list,'o')
        i+=1

    plt.show()

def visualize_image(data, idata:int = -1, title:str = ""):
    plt.title(title)
    if idata != -1:
        image = data[idata]
        image = image.reshape(3,32,32)
        print(image.shape)
        plt.imshow(image.T)
    else:

        rows = 2
        columns = 2
        fig = plt.figure(figsize=(10, 10))
        i = 1
        for img in data:
            fig.add_subplot(rows, columns, i)
            plt.imshow(img.T)
            i += 1

    plt.show()

     
        

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    data,labels = load_initial_data()
    train_loader, test_loader = data_processing(data,labels)
    net,lossfun,optimizer = create_model(False)
    net.to(device)
    epochs = 75
    losses = train_model(net,lossfunction=lossfun,optimizer=optimizer,train_loader=train_loader,device=device,epochs=epochs)
    

    img = data[0]
    img = img.reshape((1,3,32,32))
    img = torch.tensor(img).float()
    img = img.to(device)
    yHat = net(img)
    #print(yHat.shape)

    yHat = yHat.to('cpu')
    img = img.to('cpu')

    img = np.array(img.detach(), np.int32)
    img = img.reshape((3,32,32))

    yHat = np.array(yHat.detach(), np.int32)
    yHat = yHat.reshape((3,32,32))

    visualize_image([img,yHat])


if __name__=="__main__":
    main()