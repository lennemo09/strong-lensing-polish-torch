import sys, os

import matplotlib.pylab as plt
import numpy as np
import glob

from scipy import signal, interpolate
import optparse
from skimage import transform
import simulation_strong as simulation
import json

try:
    from data_augmentation import elastic_transform
except:
    print("Could not load data_augmentation")

try:
    from astropy.io import fits 
except:
    print("Could not load astropy.io.fits")

def readfits(fnfits):
    """ Read in fits file using astropy.io.fits 
    """
    hdulist = fits.open(fnfits)
    dshape = hdulist[0].shape 
    if len(dshape)==2:
        data = hdulist[0].data
    elif len(dshape)==3:
        data = hdulist[0].data[0]
    elif len(dshape)==4:
        data = hdulist[0].data[0,0]
    header = hdulist[0].header
    pixel_scale = abs(header['CDELT1'])
    num_pix = abs(header['NAXIS1'])
    return data, header, pixel_scale, num_pix


def gaussian2D( coords,  # x and y coordinates for each image.
                amplitude=1,  # Highest intensity in image.
                xo=0,  # x-coordinate of peak centre.
                yo=0,  # y-coordinate of peak centre.
                sigma_x=1,  # Standard deviation in x.
                sigma_y=1,  # Standard deviation in y.
                rho=0,  # Correlation coefficient.
                offset=0,
                rot=0):  # rotation in degrees.
    """ 2D ellipsoidal Gaussian function, including 
    rotation
    """
    x, y = coords

    rot = np.deg2rad(rot)

    x_ = np.cos(rot)*x - y*np.sin(rot)
    y_ = np.sin(rot)*x + np.cos(rot)*y

    xo = float(xo)
    yo = float(yo)

    xo_ = np.cos(rot)*xo - yo*np.sin(rot) 
    yo_ = np.sin(rot)*xo + np.cos(rot)*yo

    x,y,xo,yo = x_,y_,xo_,yo_

    # Create covariance matrix
    mat_cov = [[sigma_x**2, rho * sigma_x * sigma_y],
               [rho * sigma_x * sigma_y, sigma_y**2]]
    mat_cov = np.asarray(mat_cov)
    # Find its inverse
    mat_cov_inv = np.linalg.inv(mat_cov)

    # PB We stack the coordinates along the last axis
    mat_coords = np.stack((x - xo, y - yo), axis=-1)

    G = amplitude * np.exp(-0.5*np.matmul(np.matmul(mat_coords[:, :, np.newaxis, :],
                                                    mat_cov_inv),
                                          mat_cofords[..., np.newaxis])) + offset
    return G.squeeze()

def normalize_data(data, nbit=16):
    """ Normalize data to fit in bit range, 
    convert to specified dtype
    """
    data = data - data.min()
    data = data/data.max()

    if nbit==16:
        data *= (2**nbit-1)        
        data = data.astype(np.uint16)
    elif nbit==8:
        data *= (2**nbit-1)        
        data = data.astype(np.uint8)
    elif nbit=='float':
        data = data
        
    return data

def convolvehr(data, kernel, plotit=False, 
               rebin=4, norm=True, nbit=16, 
               noise=True, cmap='turbo'):
    """ Take input data and 2D convolve with kernel 
    
    Parameters:
    ----------
    data : ndarray 
        data to be convolved 
    kernel : ndarray 
        convolutional kernel / PSF 
    """ 
    if len(data.shape)==3:
        kernel = kernel[..., None]
        ncolor = 1
    else:
        ncolor = 3
    
    if noise:
        data_noise = data + np.random.normal(0,5,data.shape)
    else:
        data_noise = data

    # Normalize the kernel
    # kernel = kernel / np.sum(kernel)
    # print("KERNEL SHAPE", kernel.shape)

    data_convolved = signal.fftconvolve(data_noise, kernel, mode='same')

    if norm is True:
         data_convolved = normalize_data(data_convolved, nbit=nbit)
         data = normalize_data(data, nbit=nbit)

    data_residual = data_convolved - data

    data_restored = data_convolved - data_residual

    dataLR = data_convolved[rebin//2::rebin, rebin//2::rebin]

    if plotit:
        fig, axs = plt.subplots(1, 4, figsize=(18, 6))
        fig.suptitle("Convolution Process", fontsize=20)

        # Plot the original data
        axs[0].imshow(data, cmap=cmap)
        axs[0].set_title("Original Data (True)")
        axs[0].axis('off')

        # Plot the convolved data
        axs[1].imshow(data_convolved, cmap=cmap)
        axs[1].set_title("Convolved Data (Dirty)")
        axs[1].axis('off')

        # Plot the downsampled (dataLR) result
        # axs[2].imshow(dataLR, cmap=cmap)
        # axs[2].set_title("Downsampled Data (dataLR)")
        # axs[2].axis('off')

        axs[2].imshow(data_residual, cmap=cmap)
        axs[2].set_title(f"Residual [{data_residual.min():.4f},{data_residual.max():.4f}]\n(Dirty - True)")
        axs[2].axis('off')

        axs[3].imshow(data_restored, cmap=cmap)
        axs[3].set_title(f"Data restored [{data_restored.min():.4f},{data_restored.max():.4f}]\n(Dirty - Residual)")
        axs[3].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('./data_gen_debug.png')

    return dataLR, data_noise, data_convolved, data_residual

def create_LR_image(fl, kernel, fdirout=None, 
                    galaxies=False, plotit=False, 
                    norm=True, sky=False, rebin=4, nbit=16, 
                    distort_psf=False, subset='train', src_density=5,
                    nimages=800, nchan=1, save_img=True, nstart=0):
    """ Create a set of image pairs (true sky, dirty image) 
    and save to output directory 

    Parameters:
    ----------
    fl : str / list 
        Input file list 
    kernel : ndarray 
        PSF array 
    fdirout : str 
        Path to save output data to 
    galaxies : bool 
        Simulate galaxies 
    plotit : bool 
        Display plots for each image pair
    norm : bool 
        Normalize data 
    sky : bool 
        Use SKA sky data as input 
    rebin : int 
        1D resolution factor between true sky and convolved image 
    nbit : int 
        Number of bits for image data 
    distort_psr : bool 
        Distort each image pair's PSF with a difference perturbation 
    nimages : int 
        Number of image pairs 
    nchan : int 
        Number of radio frequency channels 
    save_img : bool 
        Save down images 

    Returns: 
    --------
    dataLR: ndarray 
        Convolved image arrays
    data, data_noise : ndarray 
    """
    if type(fl) is str:
        fl = glob.glob(fl+'/*.png')
        if len(fl)==0:
            print("Input file list is empty")
            exit()
    elif type(fl) is list:
        fl.sort()
    elif fl==None:
        pass
    else:
        print("Expected a list or a str as fl input")
        return

    assert subset in ['train', 'valid']

    nbit = 'float'
    
    fdiroutHR = options.fdout+'/POLISH_%s_true/'%subset
    fdiroutLR = options.fdout+'/POLISH_%s_dirty_lowres_x%d/'%(subset,rebin)
    fdiroutLR_same_res = options.fdout+'/POLISH_%s_dirty/'%(subset)
    fdiroutResidual = options.fdout+'/POLISH_%s_residual/'%subset
    fdiroutGalaxies = f"{options.fdout}/galaxy_info/"
    galaxy_info_file = f"{fdiroutGalaxies}/galaxies_{subset}.json"

    # Ensure output directory exists
    os.makedirs(fdiroutGalaxies, exist_ok=True)

    galaxy_info = {}  # Dictionary to store galaxy parameters

    for ii in range(nimages):
        if fl is not None:
            fn = fl[ii]
            if fdiroutLR is None:
                fnout = fn.strip('.png')+'-conv.npy'
            else:
                fnoutLR = fdiroutLR + fn.split('/')[-1][:-4] + 'x%d.png' % rebin
                fnoutLR_same_res = fdiroutLR_same_res + fn.split('/')[-1][:-4] + '.png'
                fnoutResidual = fdiroutResidual + fn.split('/')[-1][:-4] + '.png'
        else:
            fn = '%04d.png'%(ii+nstart)
            fnoutLR = fdiroutLR + fn[:-4] + 'x%d.png' % rebin
            fnoutLR_same_res = fdiroutLR_same_res + fn[:-4] + '.png'
            fnoutResidual = fdiroutResidual + fn[:-4] + '.png'
        if os.path.isfile(fnoutLR):
            print("File exists, skipping %s"%fnoutLR)
            continue

        if ii%10==0:
            print("Finished %d/%d" % (ii, nimages))

        if galaxies:
            Nx, Ny = NSIDE, NSIDE
            data = np.zeros([Nx,Ny])

            # Get number of sources in this simulated image
            nsrc = np.random.poisson(int(src_density*(Nx*Ny*PIXEL_SIZE**2/60.**2))) # arcmin, number of galaxies in image
            fdirgalparams = fdirout+'/galparams/'
            if not os.path.isdir(fdirgalparams):
                os.system('mkdir %s' % fdirgalparams)
            fnblobout = None#fdirgalparams + fn.split('/')[-1].strip('.png')+'GalParams.txt'
#            SimObj = simulation.SimRadioGal(nx=2*Nx, ny=2*Ny, src_density_sqdeg=src_density*3600.)
            SimObj = simulation.SimRadioGal(nx=Nx, ny=Ny, src_density_sqdeg=src_density*3600.)
            data, lensed_galaxies, non_lensed_galaxies = SimObj.sim_sky(distort_gal=False, fnblobout=fnblobout)

            galaxy_info[ii] = {
                "lensed": lensed_galaxies,
                "non_lensed": non_lensed_galaxies
            }

            if len(data.shape)==2:
                data = data[..., None]
                
            norm = True
        elif sky:
            data = np.load('SKA-fun-model.npy')
            data = data[800:800+4*118, 800:800+4*124]
            mm = np.where(data==data.max())[0]
            data[data<0] = 0
            data /= (data.max()/255.0/12.)
            data[data>255] = 255
            data = data.astype(np.uint8)
            data = data[..., None]
        
        if distort_psf:
            for aa in [1]:
                kernel_ = kernel[..., None]*np.ones([1,1,3])
#                alphad = np.random.uniform(0,5)
                alphad = np.random.uniform(0,20)
                if plotit:
                    plt.subplot(131)
                    plt.imshow(kernel,vmax=0.1,cmap='Greys')
                kernel_ = elastic_transform(kernel_, alpha=alphad,
                                           sigma=3, alpha_affine=0)
                if plotit:
                    plt.subplot(132)
                    plt.imshow(kernel_[..., 0], vmax=0.1, cmap='Greys')
                    plt.subplot(133)
                    plt.imshow(kernel-kernel_[..., 0],vmax=0.1, vmin=-0.1, cmap='Greys')
                    plt.colorbar()
                    plt.show()

                kernel_ = kernel_[..., 0]
                fdiroutPSF = fdirout[:-6]+'/psf/'
                fnout1=fdirout+'./test%0.2f.png'%aa
                fnout2=fdirout+'./test%0.2fx4.png'%aa
                #np.save(fdiroutPSF+fn.split('/')[-1][:-4] + '-%0.2f-.npy'%alphad, kernel_)
        else:
            kernel_ = kernel

        if np.random.randint(0, 100) % 5 == -1:
            datalobe = np.load('lober.npy')[:,:,None]
            nxlobe, nylobe = np.random.randint(0, len(datalobe)), np.random.randint(0, len(datalobe))
            Snu = np.random.uniform(100, 8e3)
            rotangle = np.random.randint(0, 360)
            datalobe = transform.rescale(datalobe, np.random.uniform(0.05, 1))
            datalobe = transform.rotate(datalobe, 
                                        angle=rotangle,
                                        resize=True, 
                                        center=None,
                                        order=1,
                                        mode='constant',
                                        cval=0)
            try:
                data[nxlobe:nxlobe+len(datalobe), nylobe:nylobe+len(datalobe)] += Snu*datalobe
            except:
                print("That didnt work")
                pass
            
        dataLR, data_noise, data_convolved, data_residual = convolvehr(data, kernel_, plotit=plotit, 
                                        rebin=rebin, norm=norm, nbit=nbit, 
                                        noise=True)

        nhr = len(data)
        # Select inner square with side length 1/2
        
#        data = data[int(0.25*nhr) : int(0.75*nhr), int(0.25*nhr) : int(0.75*nhr)]
#        nlr = len(dataLR)
#        dataLR = dataLR[int(0.25*nlr) : int(0.75*nlr), int(0.25*nlr) : int(0.75*nlr)]

        # print(f"Data {data.shape} before normalize: [{data.min()};{data.max()}]")
        # print(f"DataLR {dataLR.shape} before normalize: [{dataLR.min()};{dataLR.max()}]")
        # print(f"DataConvolved {data_convolved.shape} before normalize: [{data_convolved.min()};{data_convolved.max()}]")
        # print(f"DataResidual {data_residual.shape} before normalize: [{data_residual.min()};{data_residual.max()}]")

        data = normalize_data(data, nbit=nbit)
        dataLR = normalize_data(dataLR, nbit=nbit)
        data_convolved = normalize_data(data_convolved, nbit=nbit)

        # print(f"Data {data.shape} after normalize: [{data.min()};{data.max()}]")
        # print(f"DataLR {dataLR.shape} after normalize: [{dataLR.min()};{dataLR.max()}]")
        # print(f"DataConvolved {data_convolved.shape} after normalize: [{data_convolved.min()};{data_convolved.max()}]")
        # print(f"DataResidual {data_residual.shape} after normalize: [{data_residual.min()};{data_residual.max()}]")
        
        if nbit=='float':
            np.save(fnoutLR[:-4], dataLR)
            np.save(fnoutLR_same_res[:-4], data_convolved)
            np.save(fnoutResidual[:-4], data_residual)

        if galaxies or sky:
            fnoutHR = fdiroutHR + fn.split('/')[-1][:-4] + '.png'
            fnoutHRnoise = fdiroutHR + fn.split('/')[-1][:-4] + 'noise.png'

            if nbit=='float':
                np.save(fnoutHR[:-4], data)

        del dataLR, data, data_noise

        with open(galaxy_info_file, 'w') as f:
            json.dump(galaxy_info, f, indent=4)
        # print(f"Saved galaxy information to {galaxy_info_file}")
 
if __name__=='__main__':
    """
    Example run:
    python polish/make_img_pairs_strong.py -k ./psf/dsa-2000-fullband-psf.fits -o /scratch/ondemand28/len/data/DSA_1024_x2_strong/ --nside 1024 -r 2 -s 1024
    """
    parser = optparse.OptionParser(prog="hr2lr.py",
                   version="",
                   usage="%prog [OPTIONS]",
                   description="Take high resolution images, convolve them, \
                   and save output.")

    parser.add_option('-d', dest='fdirin', default=None,
                      help="input directory if high-res images already exist")
    parser.add_option('-k', '--kernel', dest='kernel', type='str',
                      help="", default='Gaussian')
    parser.add_option("-s", "--ksize", dest='ksize', type=int,
                      help="size of kernel", default=256)
    parser.add_option('-o', '--fdout', dest='fdout', type='str',
                      help="output directory", default='./')
    parser.add_option('-p', '--plotit', dest='plotit', action="store_true",
                      help="plot")
    parser.add_option('-x', '--galaxies', dest='galaxies', action="store_true",
                      help="only do point sources", default=True)
    parser.add_option('--sky', dest='sky', action="store_true",
                      help="use SKA mid image as input")
    parser.add_option('-r', '--rebin', dest='rebin', type=int,
                      help="factor to spatially rebin", default=4)
    parser.add_option('-b', '--nbit', dest='nbit', type=int,
                      help="number of bits for image", default=16)
    parser.add_option('-n', '--nchan', dest='nchan', type=int,
                      help="number of frequency channels for image", default=1)
    parser.add_option('--ntrain', dest='ntrain', type=int,
                      help="number of training images", default=800)
    parser.add_option('--nvalid', dest='nvalid', type=int,
                      help="number of validation images", default=100)
    parser.add_option('--distort_psf', dest='distort_psf', action="store_true",
                      help="perturb PSF for each image generated")
    parser.add_option('--pix', dest='pixel_size', type=float, default=0.25,
                      help="pixel size of true sky in arcseconds")
    parser.add_option('--src_density', dest='src_density', type=float, default=5,
                      help="source density per sq arcminute")
    parser.add_option('--nside', dest='nside', type=int, default=2048,
                      help="dimension of HR image")

    # Frequency range in GHz
    FREQMIN, FREQMAX = 0.7, 2.0

    options, args = parser.parse_args()
    PIXEL_SIZE = options.pixel_size
    src_density = options.src_density
    NSIDE = options.nside

    # Read in kernel. If -k is not given, assume Gaussian kernel 
    if options.kernel.endswith('npy'):
        kernel = np.load(options.kernel)
        nkern = len(kernel)
        kernel = kernel[nkern//2-options.ksize//2:nkern//2+options.ksize//2, 
                        nkern//2-options.ksize//2:nkern//2+options.ksize//2]
    elif options.kernel in ('Gaussian', 'gaussian'):
        kernel1D = signal.gaussian(8, std=1).reshape(8, 1)
        kernel = np.outer(kernel1D, kernel1D)
    elif options.kernel.endswith('fits'):
        kernel, header, pixel_scale_psf, num_pix = readfits(options.kernel)
        nkern = len(kernel)

        kernel = kernel[nkern//2-options.ksize//2:nkern//2+options.ksize//2, 
                        nkern//2-options.ksize//2:nkern//2+options.ksize//2]
        
        pixel_scale_psf *= 3600
        if abs((1-pixel_scale_psf/PIXEL_SIZE)) > 0.025:
            print("Stretching PSF by %0.3f to match map" % (pixel_scale_psf/PIXEL_SIZE))
            kernel = transform.rescale(kernel, pixel_scale_psf/PIXEL_SIZE)

        

    # Input directory
    if options.fdirin is None:
        fdirinTRAIN = None
        fdirinVALID = None 
    else:
        fdirinTRAIN = options.fdirin+'/POLISH_train_true/'
        fdirinVALID = options.fdirin+'/POLISH_valid_true/'

    # Output directories for training and validation. 
    # If they don't exist, create them
    fdiroutTRAIN_HR = options.fdout+'/POLISH_train_true'
    fdiroutVALID_HR = options.fdout+'/POLISH_valid_true'
    fdiroutTRAIN_LR_same_res = options.fdout+'/POLISH_train_dirty'
    fdiroutVALID_LR_same_res = options.fdout+'/POLISH_valid_dirty'
    fdiroutTRAIN_LR = options.fdout+'/POLISH_train_dirty_lowres_x%d'%options.rebin
    fdiroutVALID_LR = options.fdout+'/POLISH_valid_dirty_lowres_x%d'%options.rebin
    fdiroutTRAIN_residual = options.fdout+'/POLISH_train_residual'
    fdiroutVALID_residual = options.fdout+'/POLISH_valid_residual'
    
    fdiroutPSF = options.fdout+'/psf/'

    if not os.path.isdir(fdiroutTRAIN_HR):
        print("Making output training true sky directory")
        os.system('mkdir -p %s' % fdiroutTRAIN_HR)

    if not os.path.isdir(fdiroutTRAIN_LR):
        print("Making output training dirty downscaled directory")
        os.system('mkdir -p %s' % fdiroutTRAIN_LR)

    if not os.path.isdir(fdiroutTRAIN_LR_same_res):
        print("Making output training dirty original size directory")
        os.system('mkdir -p %s' % fdiroutTRAIN_LR_same_res)

    if not os.path.isdir(fdiroutVALID_HR):
        print("Making output validation true sky directory")
        os.system('mkdir -p %s' % fdiroutVALID_HR)

    if not os.path.isdir(fdiroutVALID_LR):
        print("Making output validation dirty downscaled directory")
        os.system('mkdir -p %s' % fdiroutVALID_LR)

    if not os.path.isdir(fdiroutVALID_LR_same_res):
        print("Making output validation dirty original size directory")
        os.system('mkdir -p %s' % fdiroutVALID_LR_same_res)

    if not os.path.isdir(fdiroutTRAIN_residual):
        print("Making output training residual directory")
        os.system('mkdir -p %s' % fdiroutTRAIN_residual)

    if not os.path.isdir(fdiroutVALID_residual):
        print("Making output validation residual directory")
        os.system('mkdir -p %s' % fdiroutVALID_residual)

    if not os.path.isdir(fdiroutPSF):
        print("Making output PSF directory")
        os.system('mkdir -p %s' % fdiroutPSF)

    print("saving idealized PSF")
    np.save('%s/psf_ideal.npy' % fdiroutPSF, kernel)

    # Create image pairs for training
    create_LR_image(fdirinTRAIN, kernel, fdirout=options.fdout, 
            plotit=options.plotit, galaxies=options.galaxies, 
                    sky=options.sky, rebin=options.rebin, nbit=options.nbit, src_density=src_density,
            distort_psf=options.distort_psf, nchan=options.nchan, subset='train',
            nimages=options.ntrain, nstart=0)   
    # Create image pairs for validation set
    create_LR_image(fdirinVALID, kernel, fdirout=options.fdout, 
            plotit=options.plotit, galaxies=options.galaxies, 
                    sky=options.sky, rebin=options.rebin, nbit=options.nbit, src_density=src_density,
            distort_psf=options.distort_psf, nchan=options.nchan, subset='valid',
            nimages=options.nvalid, nstart=options.ntrain)





