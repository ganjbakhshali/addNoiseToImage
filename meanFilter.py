import cv2
import numpy as np
from matplotlib import pyplot as plt

def Averaging(path):
    img = cv2.imread(path)
    filterSize=35
    kernel = np.ones((filterSize,filterSize),np.float32)/filterSize**2
    dst = cv2.filter2D(img,-1,kernel)
    # plt.subplot(121),plt.imshow(img),plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    fileName='Averaging_'+path
    cv2.imwrite(fileName, dst)
def gaussianNoise(path):
    img = cv2.imread(path)
    row,col,ch= img.shape
    mean = 0
    var = 0.9
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img + gauss
    fileName='gaussianNoise_'+path
    cv2.imwrite(fileName, noisy)

def SaltAndPapaer(path):
    image = cv2.imread(path)
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.04
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[coords] = 0
    nameFile='SaltAndPapaer_'+path
    cv2.imwrite(nameFile, out)

if __name__ == "__main__":
    for i in range(6):
        p=i+1
        path=str(p)+'.jpg'
        Averaging(path)
        gaussianNoise(path)
        SaltAndPapaer(path)


    
