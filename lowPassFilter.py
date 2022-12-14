from skimage import io
import matplotlib.pyplot as plt
from scipy import fftpack
import numpy as np

#Open image
image = io.imread('./EnglishImg/English/Img/noisyImag/Sample005/img005-00002.png', as_gray=True)
M,N = image.shape
f, ax = plt.subplots(figsize=(5,5))
ax.imshow(image,cmap = "gray")
ax.set_title('Original Noisy Image')

#Compute discrete Fourier transform.
F = fftpack.fftn(image) 
F_magnitude = np.abs(F)   

#shift frequency component
F_magnitude = fftpack.fftshift(F_magnitude)

# Low pass filter the frequency
K = 30
F_magnitude[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0
peaks = F_magnitude > np.percentile(F_magnitude, 98) #for 98 percentile
peaks1 = F_magnitude > np.percentile(F_magnitude, 100) #for 100 percentile

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

#Display denoised image and noisy image
f, (ax1, ax3) = plt.subplots(1, 2, figsize=(10,10))
ax1.imshow(image_filtered, cmap="gray")
ax1.set_title('Reconstructed image(98 %ile)');
ax3.imshow(image_filtered1, cmap="gray")
ax3.set_title('Reconstructed image(100 %ile)');
plt.show()