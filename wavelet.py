import matplotlib.pyplot as plt
import skimage.io
from skimage.restoration import denoise_wavelet, cycle_spin
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio

"""This file provide a script to test the Wavelet domain filtering on a input image.
   This script automatically add the noise to you image.
   You just have to enter your local path line """
#import image
original= skimage.io.imread('./EnglishImg/English/Img/GoodImg/Bmp//Sample005/img005-00002.png')
original=skimage.img_as_float(original)

#Add noise to image
noisy = random_noise(original, mean=0.3)


#Display images
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                       sharex=False, sharey=False)

ax = ax.ravel()
psnr_noisy = peak_signal_noise_ratio(original, noisy)
ax[0].imshow(noisy)
ax[0].axis('off')
ax[0].set_title(f'Noisy\nPSNR={psnr_noisy:0.4g}')

#denoise images
denoise_kwargs = dict(channel_axis=-1, convert2ycbcr=True, wavelet='db1',
                      rescale_sigma=True)
im_bayescs = cycle_spin(noisy, func=denoise_wavelet, max_shifts=3,
                        func_kw=denoise_kwargs, channel_axis=-1)

#Display denoised image
ax[1].imshow(im_bayescs)
ax[1].axis('off')
psnr = peak_signal_noise_ratio(original, im_bayescs)
ax[1].set_title(
    f'Denoised: 4X4 shifts\nPSNR={psnr:0.4g}')

plt.show()
