import time
import os 
import random
import cv2
import torch 
from torch import nn 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.restoration import denoise_wavelet, cycle_spin
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import PIL
import torchvision.transforms as T
from torchsummary import summary

#constant definition
BATCH_SIZE=130     
EPOCHS=25
LEARNING_RATE=0.003


class NeuralNet(nn.Module):
    def __init__(self) :
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining a 3rd 2D convolution layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1 * 16 * 16, 62),
        )
    
    def forward(self,x):
        x=self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def addNoise(img):
    # transform= T.ToPILImage()
    # img=transform(img)
    img=np.transpose(img.numpy(), (1, 2, 0))
    img=skimage.util.random_noise(img, mode='gaussian', mean=0.3,seed=None, clip=True)
    img=skimage.util.random_noise(img, mode='pepper', amount=0.1,seed=None, clip=True)
    return img

def mean_filter(image, height, width): #function of the mean filter.
    kernel = get_kernel()
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            # We get the area to be filtered with range indexing.
            area = image[row - 1:row + 2, column - 1:column + 2]
            image[row][column] = np.sum(np.multiply(kernel,area))
    return image

def median_filter(image, height, width): #function of the median filter.
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            area = image[row - 1:row + 2, column - 1:column + 2]
            image[row][column] = np.median(area)
    return image

def waveletTransform(original):
    original=np.transpose(original.numpy(), (1, 2, 0))
    #Add noise to image
    noisy = random_noise(original, mean=0.3)

    #denoise images
    denoise_kwargs = dict(channel_axis=-1, convert2ycbcr=True, wavelet='db1',
                        rescale_sigma=True)
    im_bayescs = cycle_spin(noisy, func=denoise_wavelet, max_shifts=3,
                            func_kw=denoise_kwargs, channel_axis=-1)
    return im_bayescs
def domainFiltering(image,M,N):
    #Compute discrete Fourier transform.
    F = fftpack.fftn(image) 
    F_magnitude = np.abs(F)   

    #shift frequency component
    F_magnitude = fftpack.fftshift(F_magnitude)

    # Low pass filter the frequency
    K = 30
    F_magnitude[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0
    peaks = F_magnitude > np.percentile(F_magnitude, 96) #for 98 percentile
    peaks1 = F_magnitude > np.percentile(F_magnitude, 98) #for 100 percentile

    # Shift peaks back to original position
    peaks = fftpack.ifftshift(peaks) 
    peaks1 = fftpack.ifftshift(peaks1)

    # Make a copy of the original spectrum
    F_dim = F.copy() 
    F_dim1 = F.copy() 

    # Set those peak coefficients to zero
    F_dim = F_dim * peaks.astype(int) 
    F_dim1 = F_dim1 * peaks1.astype(int) 

    # Inverse fourier transform
    image_filtered = np.real(fftpack.ifft2(F_dim)) 
    image_filtered1 = np.real(fftpack.ifft2(F_dim1))
    return image_filtered1

def combined_filter(image, height, width): #function of the combined filter with the ratio for the median and mean filter.
    for row in range(1, height + 1):
        for column in range(1, width + 1):
            area = image[row - 1:row + 2, column - 1:column + 2]
            mean_filter = np.sum(np.multiply(get_kernel(),area))
            median_filter = np.median(area)
            image[row][column] = 0.2 * mean_filter + (1 - 0.2) * median_filter
    return image

#Function to import dataset
def downloadDataSets() :
    images=[]
    group=0
    folder_dir = "./EnglishImg/English/Img/GoodImg/Bmp/"    #Path to one type of images
    for samples in os.listdir(folder_dir):
        sub_folder_dir=f"{folder_dir}{samples}"
        #print(sub_folder_dir)
        for image in os.listdir(sub_folder_dir):
            if (image.endswith(".png")):
                #Open image
                originalImage=cv2.imread(f'{sub_folder_dir}/{image}')
                #Reshape image
                originalImage=cv2.resize(originalImage, (64, 64))
                #Convert to tensor
                convert_tensor = ToTensor()
                originalImage=convert_tensor(originalImage)
                images.append((originalImage,group))
        group=group+1

    print(f"{len(images)} images in the dataset")
    #Split dataset
    random.shuffle(images)
    trainData=images[:round(0.9*len(images))]
    validationData=images[round(0.9*len(images)):]
    print(f"{len(trainData)} images in train images")
    print(f"{len(validationData)} images in validation images")
    for element in validationData:
        convert_tensor = ToTensor()
        #add noise
        #element=(addNoise(element[0]),element[1])
        #denoise
        denoiseImg=waveletTransform(element[0])
        element=(convert_tensor(denoiseImg),element[1])

    return trainData, validationData

def trainOneEpach(model, dataLoader, lossFunction, optimiser, device, losses):
    for inputs, targets in dataLoader:
        inputs, targets=inputs.to(device), targets.to(device)
        
        #loss 
        predictions=model(inputs)
        loss=lossFunction(predictions,targets)

        #propagate loss and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print (f"loss:{loss.item()}")
    losses.append(loss.item())
    return losses
    
def trainModel(model, dataLoader, lossFunction, optimiser, device, epochs):
    losses = []
    for i in range (epochs):
        print(f"epoch : {i+1}")
        losses=trainOneEpach(model, dataLoader, lossFunction, optimiser, device, losses)
        print("- - - - - ")
    print('training complete')
    plt.plot(np.array(losses), 'r')
    plt.show()

def showImage(dataLoader):
    # get images
    data = iter(dataLoader)
    images ,_= data.next()
    # display images
    img=np.transpose(images[0].numpy(), (1, 2, 0))
    plt.imshow(img)
    plt.show()
    img2=np.transpose(images[1].numpy(), (1, 2, 0))
    plt.imshow(img2)
    plt.show()


classMapping=[
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    
]

def predict(model,input,target,classMapping):
    model.eval()
    with torch.no_grad():
        predictions=model(input)
        #tensor (nb samples 1,nc classes 10)
        predicted=[]
        expected=[]
        for i in range (len(predictions)):
            predictedIndex=predictions[i].argmax(0)
            predicted.append(classMapping[predictedIndex])
            expected.append(classMapping[target[i]])
    return predicted, expected

def get_kernel(): # the declaration of the kernel for the mean filter.
    return np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])


if __name__ == "__main__":
    
    trainData,validationData=downloadDataSets()   #download the data
    print("dataSet downloaded")

    #Add noise to images in validation dataset

    #Denoise images in validation dataseet
    trainDataLoader=DataLoader(trainData,batch_size=BATCH_SIZE) #Load the data
    testDataLoader=DataLoader(validationData,batch_size=BATCH_SIZE) #Load the data


    #build the model
    neuralNetwork=NeuralNet().to('cpu')
    #summary(neuralNetwork, (28, 28))

    #Display the first two images
    # showImage(trainDataLoader)
    # showImage(testDataLoader)

    # --------------------------------train the model---------------------------------------
    lossFunc=nn.CrossEntropyLoss()
    optimiser=torch.optim.Adam(neuralNetwork.parameters(),lr=LEARNING_RATE)
    trainModel(neuralNetwork,trainDataLoader,lossFunc,optimiser,'cpu',EPOCHS)

    #---------------------------------save the training-------------------------------
    torch.save(neuralNetwork.state_dict(),f'./weights/training10.pth')

    print("model stored")

    


    #------------------------test all the images in the test dataset----------------------------
    totalRight=0
    start = time.process_time()
    times=[]
    for inputs, targets in testDataLoader:
        #make an inference
        predicted, expected = predict(neuralNetwork,inputs,targets,classMapping)

        for i in range (len(predicted)):
            if predicted[i]==expected[i] :
                totalRight+=1
        # if(sample%100==0):
        #     times.append(time.process_time()-start)

    print(f"'accuracy = {totalRight/len(validationData)}'%")
    #Plot the inference time
    # x = np.arange(0, 10000, 100)
    # plt.plot(x,np.array(times), 'r')
    # plt.xlabel("Number of images tested")
    # plt.ylabel('Time of execution in s')
    # plt.show()