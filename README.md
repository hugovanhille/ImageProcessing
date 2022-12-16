# Image Processing Project
![Test](image.png) 

This repository hold the code we used for our project of VE556, Image Processing course at SJTU (FAll semester)


## Dataset
You can download the datasets we used on this website
http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

## Add noise
The file [noise.py](https://github.com/hugovanhille/ImageProgressing/noise.py)  include the code we used to add noise to a dataset present in local folders

## Denoising method 

The file [Imae_Processing.ipynb](https://github.com/hugovanhille/ImageProgressing/Imae_Processing.ipynb) include the script used to test the methods Mean Filter, Median Filter and Combined Filter.

The file [lowPassFilter.py](https://github.com/hugovanhille/ImageProgressing/lowPassFilter.py) include the script used to test the method Spatial frequency filtering.

The file [wavelet.py](https://github.com/hugovanhille/ImageProgressing/wavelet.py) include the script used to test the method wavelet domain filtering.

## Metrics Performance
In the file [Imae_Processing.ipynb](https://github.com/hugovanhille/ImageProgressing/Imae_Processing.ipynb), we have the main function that noise, denoise and calculate all the metrics for a given number of images (here 10). This function takes 2 parameters: the name of the filter(mean_filter, median_filter, combined_filter, LowPass_Filter and wavelet) and the class (0= Mean/Median/Combined, 1=Low Pass and other is Wavelet).  This function return the original images, the denoise images and all the error (RMSE, NRMSE, SNR and PSNR). 

For example, you could do: image, denoise_image, rmse, nrmse, snr, psnr=main(mean_filter, 0).

At the end of this file, we plot the result for 10 images for all the different filter.

## Deep Neural Network (DNN)
The file [trainNN.py](https://github.com/hugovanhille/ImageProgressing/trainNN.py) contain the code to ttrain the NN we used as well as the code to do inferences on dataset in order to determine the accuracy of the validation dataset after denoising by the different methods.

## _Authors_

This project has been conducted by :

* **Hugo Vanhille** _alias_ [@hugovanhille](https://github.com/hugovanhille)
* **Maxence Vandendorpe** _alias_ [@altreon100](https://github.com/altreon100)

