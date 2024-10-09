from torchvision import transforms
import numpy as np
import albumentations as A


import os
import time
from glob import glob

import torch
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.feature_extraction import image
from tqdm import tqdm

from IPython.display import clear_output
from tifffile import imsave


def set_global_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(42)
import albumentations as A
import numpy as np
import torch

class Augmentations:
    def __init__(self):
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(p= 0.5, limit=(90, 180) )
        ], additional_targets={'image0': 'image', 'noisy_image_1': 'image', 'noisy_image_2': 'image'})   
    

    def __call__(self, *args, **kwargs):
        return self.transforms(*args,**kwargs)



def normalize(img, mean, std):
    """Normalize an array of images with mean and standard deviation.
        Parameters
        ----------
        img: array
            An array of images.
        mean: float
            Mean of img array.
        std: float
            Standard deviation of img array.
        """
    return (img - mean) / std


def denormalize(img, mean, std):
    """Denormalize an array of images with mean and standard deviation.
    Parameters
    ----------
    img: array
        An array of images.
    mean: float
        Mean of img array.
    std: float
        Standard deviation of img array.
    """
    return (img * std) + mean


def convertToFloat32(train_images, val_images):
    """Converts the data to float 32 bit type.
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
        """
    x_train = train_images.astype('float32')
    x_val = val_images.astype('float32')
    return x_train, x_val


def getMeanStdData(train_images, val_images):
    """Compute mean and standrad deviation of data.
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
    """
    x_train_ = train_images.astype('float32')
    x_val_ = val_images.astype('float32')
    data = np.concatenate((x_train_, x_val_), axis=0)
    mean, std = np.mean(data), np.std(data)
    return mean, std


def convertNumpyToTensor(numpy_array):
    """Convert numpy array to PyTorch tensor.
    Parameters
    ----------
    numpy_array: numpy array
        Numpy array.
    """
    return torch.from_numpy(numpy_array)


def preprocess(train_patches, val_patches):
    data_mean, data_std = getMeanStdData(train_patches, val_patches)
    x_train, x_val = convertToFloat32(train_patches, val_patches)
    x_train_extra_axis = x_train[:, np.newaxis]
    x_val_extra_axis = x_val[:, np.newaxis]
    x_train_tensor = convertNumpyToTensor(x_train_extra_axis)
    x_val_tensor = convertNumpyToTensor(x_val_extra_axis)
    return x_train_tensor, x_val_tensor, data_mean, data_std


def get_trainval_patches(x, split_fraction=0.85, augment=True, patch_size=128, num_patches=None):
    np.random.shuffle(x)
    train_images = x[:int(0.85 * x.shape[0])]
    val_images = x[int(0.85 * x.shape[0]):]
    if (augment):
        train_images = augment_data(train_images)
    x_train_crops = extract_patches(train_images, patch_size, num_patches)
    x_val_crops = extract_patches(val_images, patch_size, num_patches)
    print("Shape of training patches:", x_train_crops.shape, "Shape of validation patches:", x_val_crops.shape)
    return x_train_crops, x_val_crops


def extract_patches(x, patch_size, num_patches):
    """Deterministically extract patches from array of images.
    Parameters
    ----------
    x: numpy array
        Array of images.
    patch_size: int
        Size of patches to be extracted from each image.
    num_patches: int
        Number of patches to be extracted from each image.
    """
    img_width = x.shape[2]
    img_height = x.shape[1]
    if (num_patches is None):
        num_patches = int(float(img_width * img_height) / float(patch_size**2) * 2)
    patches = np.zeros(shape=(x.shape[0] * num_patches, patch_size, patch_size))

    for i in tqdm(range(x.shape[0])):
        patches[i * num_patches:(i + 1) * num_patches] = image.extract_patches_2d(x[i], (patch_size, patch_size),
                                                                                  num_patches,
                                                                                  random_state=i)
    return patches


def augment_data(X_train):
    """Augment data by 8-fold with 90 degree rotations and flips.
    Parameters
    ----------
    X_train: numpy array
        Array of training images.
    """
    X_ = X_train.copy()
    X_train_aug = np.concatenate((X_train, np.rot90(X_, 1, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))
    return X_train_aug


def loadImages(path):
    """Load images from a given directory.
    Parameters
    ----------
    path: String
        Path of directory from where to load images from.
    """
    files = sorted(glob(path))
    data = []
    print(path)
    for f in files:
        if '.png' in f:
            im_b = np.array(io.imread(f))
        if '.npy' in f:
            im_b = np.load(f)
        data.append(im_b)

    data = np.array(data).astype(np.float32)
    return data


def getSamples(vae, size=20, zSize=64, mu=None, logvar=None, samples=1, tq=False):
    """Generate synthetic samples from Disentangle network.
    Parameters
    ----------
    vae: VAE Object
        Disentangle model.
    size: int
        Size of generated image in the bottleneck.
    zSize: int
        Dimension of latent space for each pixel in bottleneck.
    mu: PyTorch tensor
        latent space mean tensor.
    logvar: PyTorch tensor
        latent space log variance tensor.
    samples: int
        Number of synthetic samples to generate.
    tq: boolean
        If tqdm should be active or not to indicate progress.
    """
    if mu is None:
        mu = torch.zeros(1, zSize, size, size).cuda()
    if logvar is None:
        logvar = torch.zeros(1, zSize, size, size).cuda()

    results = []
    for i in tqdm(range(samples), disable=not tq):
        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)
        recon_cpu = recon.cpu()
        recon_numpy = recon_cpu.detach().numpy()
        recon_numpy.shape = (recon_numpy.shape[-2], recon_numpy.shape[-1])
        results.append(recon_numpy)
    return np.array(results)


def interpolate(
    vae,
    z_start,
    z_end,
    steps,
    display,
    vmin=0,
    vmax=255,
):
    results = []
    for i in range(steps):
        alpha = (i / (steps - 1.0))
        z = z_end * alpha + z_start * (1.0 - alpha)
        recon = vae.decode(z)
        recon_cpu = recon.cpu()
        recon_numpy = recon_cpu.detach().numpy()
        recon_numpy.shape = (recon_numpy.shape[-2], recon_numpy.shape[-1])
        if display:
            clear_output(wait=True)
            plt.imshow(recon_numpy, vmin=vmin, vmax=vmax)
            plt.show()
        time.sleep(0.4)
        results.append(recon_numpy)
    return results


def tiledMode(im, ps, overlap, display=True, vmin=0, vmax=255, initBW=200, minBW=100, reduce=0.9):
    means = np.zeros(im.shape[1:])
    xmin = 0
    ymin = 0
    xmax = ps
    ymax = ps
    ovLeft = 0
    while (xmin < im.shape[2]):
        ovTop = 0
        while (ymin < im.shape[1]):
            inputPatch = im[:, ymin:ymax, xmin:xmax]
            a = findMode(inputPatch, initBW, minBW, reduce)
            a = a[:a.shape[0], :a.shape[1]]
            means[ymin:ymax, xmin:xmax][ovTop:, ovLeft:] = a[ovTop:, ovLeft:]

            ymin = ymin - overlap + ps
            ymax = ymin + ps
            ovTop = overlap // 2

        ymin = 0
        ymax = ps
        xmin = xmin - overlap + ps
        xmax = xmin + ps
        ovLeft = overlap // 2

        if display:
            plt.imshow(means, vmin=vmin, vmax=vmax)
            plt.show()
            clear_output(wait=True)

    return means


def findClosest(samples, q):
    """Find closest sample to a given sample.
    Parameters
    ----------
    samples: array
        Array of samples from which the closest image needs to be found.
    q: image(array)
        Image to which the closest image needs to be found.
    """
    dif = np.mean(np.mean((samples - q)**2, -1), -1)
    return samples[np.argmin(dif)]


def findMode(samples, initBW=200, minBW=100, reduce=0.9):
    """Find the modes of a distribution of images.
    Parameters
    ----------
    samples: array
        Array of samples from which the modes need to be found.
    initBW: int
        Initial bandwidth.
    minBW: int
        Minimum bandwidth.
    reduce: float
        Factor by which to reduce bandwith i n iterations.
    """
    imagesC = samples.copy()
    imagesC.shape = (samples.shape[0], samples.shape[1] * samples.shape[2])
    seed = np.mean(imagesC, axis=0)[np.newaxis, ...]
    bw = initBW
    for i in range(15):

        clustering = MeanShift(bandwidth=bw, seeds=seed, cluster_all=True).fit(imagesC)
        centers = clustering.cluster_centers_.copy()
        seed = centers
        bw = bw * reduce
        if bw < minBW:
            break

    result = seed[0]
    result.shape = (samples.shape[1], samples.shape[2])
    return result


def plotProbabilityDistribution(signalBinIndex, histogramNoiseModel, gaussianMixtureNoiseModel, device):
    """Plots probability distribution P(x|s) for a certain ground truth signal.
       Predictions from both Histogram and GMM-based Noise models are displayed for comparison.
        Parameters
        ----------
        signalBinIndex: int
            index of signal bin. Values go from 0 to number of bins (`n_bin`).
        histogramNoiseModel: Histogram based noise model
        gaussianMixtureNoiseModel: GaussianMixtureNoiseModel
            Object containing trained parameters.
        device: GPU device
        """
    max_signal = histogramNoiseModel.maxv.item()
    min_signal = histogramNoiseModel.minv.item()
    n_bin = int(histogramNoiseModel.bins.item())

    histBinSize = (max_signal - min_signal) / n_bin
    querySignal_numpy = (signalBinIndex / float(n_bin) * (max_signal - min_signal) + min_signal)
    querySignal_numpy += histBinSize / 2
    querySignal_torch = torch.from_numpy(np.array(querySignal_numpy)).float().to(device)

    queryObservations_numpy = np.arange(min_signal, max_signal, histBinSize)
    queryObservations_numpy += histBinSize / 2
    queryObservations = torch.from_numpy(queryObservations_numpy).float().to(device)
    pTorch = gaussianMixtureNoiseModel.likelihood(queryObservations, querySignal_torch)
    pNumpy = pTorch.cpu().detach().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel('Observation Bin')
    plt.ylabel('Signal Bin')
    histogram = histogramNoiseModel.fullHist.cpu().numpy()
    plt.imshow(histogram**0.25, cmap='gray')
    # plt.axhline(y=signalBinIndex + 0.5, linewidth=5, color='blue', alpha=0.5)

    plt.subplot(1, 2, 2)
    histobs = histogramNoiseModel.likelihood(queryObservations, querySignal_torch).cpu().numpy()
    # histobs_repeated = np.repeat(histobs, 2)
    # queryObservations_repeated = np.repeat(queryObservations_numpy, 2)
    plt.plot(queryObservations_numpy,
             histobs,
             label='Hist : ' + ' signal = ' + str(np.round(querySignal_numpy, 2)),
             color='blue',
             marker='.',
             linewidth=2)

    plt.plot(queryObservations_numpy,
             pNumpy,
             label='GMM : ' + ' signal = ' + str(np.round(querySignal_numpy, 2)),
             marker='.',
             color='red',
             linewidth=2)
    plt.xlabel('Observations (x) for signal s = ' + str(querySignal_numpy))
    plt.ylabel('Probability Density')
    plt.title("Probability Distribution P(x|s) at signal =" + str(querySignal_numpy))
    plt.legend()
    return {'gmm': {'x': queryObservations_numpy, 'p': pNumpy}, 'hist': {'x': queryObservations_numpy, 'p': histobs}}


def predict_mmse(vae, img, samples, device, returnSamples=False, tq=True):
    '''
    Predicts MMSE estimate.
    Parameters
    ----------
    vae: VAE object
        Disentangle model.
    img: array
        Image for which denoised MMSE estimate needs to be computed.
    samples: int
        Number of samples to average for computing MMSE estimate.
    returnSamples:
        Should the method also return the individual samples?
    tq:
        Should progress bar be shown.
    tta:
        Should test time augmentation be enabled.
    '''
    img_height, img_width = img.shape[0], img.shape[1]
    imgT = torch.Tensor(img.copy())
    image_sample = imgT.view(1, 1, img_height, img_width).to(device)
    vae.num_samples = samples
    all_samples = np.array(vae(image_sample, tqdm_bar=tq))
    samples_array = all_samples[:, 0, 0, :, :]
    if returnSamples:
        return np.mean(samples_array, axis=0), samples_array
    else:
        return np.mean(samples_array, axis=0)


def normalize_minmse(x, target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov = np.cov(x.flatten(), target.flatten())
    alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
    beta = target.mean() - alpha * x.mean()
    return alpha * x + beta


def tta_forward(x):
    """
    Augments x 8-fold: all 90 deg rotations plus lr flip of the four rotated versions.

    Parameters
    ----------
    x: data to augment

    Returns
    -------
    Stack of augmented x.
    """
    x_aug = [x, np.rot90(x, 1), np.rot90(x, 2), np.rot90(x, 3)]
    x_aug_flip = x_aug.copy()
    for x_ in x_aug:
        x_aug_flip.append(np.fliplr(x_))
    return x_aug_flip


def tta_backward(x_aug):
    """
    Inverts `tta_forward` and averages the 8 images.

    Parameters
    ----------
    x_aug: stack of 8-fold augmented images.

    Returns
    -------
    average of de-augmented x_aug.
    """
    x_deaug = [
        x_aug[0],
        np.rot90(x_aug[1], -1),
        np.rot90(x_aug[2], -2),
        np.rot90(x_aug[3], -3),
        np.fliplr(x_aug[4]),
        np.rot90(np.fliplr(x_aug[5]), -1),
        np.rot90(np.fliplr(x_aug[6]), -2),
        np.rot90(np.fliplr(x_aug[7]), -3)
    ]
    return np.mean(x_deaug, 0)


def predict_and_save(img, vae, num_samples, device, fraction_samples_to_export, export_mmse, export_results_path, tta):
    '''
    Predict denoised images and save results to disk.
    Parameters
    ----------
    img: array or list
        A stack of tif images.
    vae: Disentangle model
    num_samples: int
        Number of samples to generate and use for computing MMSE.
    device: cuda device or cpu
    fraction_samples_to_export: float between 0 (inclusive) and 1 (inclusive)
        Number of samples to save on disk for each noisy image.
    export_mmse: bool
        Should MMSE estimate also be exported?
    export_results_path: str
        path where all results will be exported.
    tta: bool
        Use test-time augmentation if set to True.

    '''
    mmse_results = []
    if isinstance(img, (list)):
        num_images = len(img)
    if isinstance(img, (np.ndarray)):
        num_images = img.shape[0]
    for i in range(num_images):
        print("Processing image:", i)
        if tta:
            aug_imgs = tta_forward(img[i])
            mmse_aug = []
            for j in range(len(aug_imgs)):
                if (j == 0):
                    mmse, samples = predict_mmse(vae, aug_imgs[j], num_samples, device=device, returnSamples=True)
                else:
                    mmse = predict_mmse(vae, aug_imgs[j], num_samples, device=device, returnSamples=False)
                mmse_aug.append(mmse)

            mmse_back_transformed = tta_backward(mmse_aug)
            mmse_results.append(mmse_back_transformed)
        else:
            mmse, samples = predict_mmse(vae, img[i], num_samples, device=device, returnSamples=True)
            mmse_results.append(mmse)
        if fraction_samples_to_export > 0:
            subdir = export_results_path + "/" + str(i).zfill(3) + "/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            imsave(subdir + "samples_for_image_" + str(i).zfill(3) + ".tif",
                   np.array(samples[:int(num_samples * fraction_samples_to_export)]).astype("float32"))
    if (export_mmse):
        imsave(export_results_path + "/mmse_results.tif", np.array(mmse_results).astype("float32"))
    return mmse_results


def plot_qualitative_results(noisy_input, vae, device):
    '''
    Plot qualitative results on patches.
    Parameters
    ----------
    noisy_input: array or list
        A stack of tif images.
    '''

    for j in range(5):

        # we select a random crop
        size_uncropped = int(0.14 * (np.minimum(noisy_input[0].shape[0], noisy_input[0].shape[1])))
        size = size_uncropped - (size_uncropped % (2**vae.n_depth))
        minx = np.random.randint(0, noisy_input[0].shape[0] - size)
        miny = np.random.randint(0, noisy_input[0].shape[1] - size)
        img = noisy_input[0][minx:minx + size, miny:miny + size]

        # generate samples and MMSE estimate
        imgMMSE, samps = predict_mmse(vae, img, samples=100, device=device, returnSamples=True)

        plt.figure(figsize=(20, 6.75))

        # We display the noisy input image
        ax = plt.subplot(1, 6, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(img, cmap='magma')
        plt.title('input')

        # We display the average of 100 predicted samples
        ax = plt.subplot(1, 6, 6)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(imgMMSE, cmap='magma')
        plt.title('MMSE (100 samples)')

        # We also display the first 4 samples
        for i in range(4):
            ax = plt.subplot(1, 6, i + 2)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(samps[i], cmap='magma')
            plt.title('prediction ' + str(i + 1))

        plt.show()