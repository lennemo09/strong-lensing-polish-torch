import matplotlib.pylab as plt
import numpy as np

from astropy.modeling.models import Sersic2D
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
# from numba import jit

try:
    from data_augmentation import elastic_transform
except:
    print("Could not load data_augmentation")

class SimRadioGal:
    """ Class for simulations the microJansky radio sky 
    """
    def __init__(self, 
                nx=2000,
                ny=2000, 
                pixel_size=0.25,
                nchan=1,
                src_density_sqdeg=13000, 
                freqmin=0.7,
                freqmax=2.0):
        self._nx = nx
        self._ny = ny
        self._pixel_size = pixel_size
        self._nchan = nchan
        self._src_density_sqdeg = src_density_sqdeg
        self.nblock=250
        self._freqmin, self._freqmax = freqmin, freqmax

    def galparams(self):
        # Choose random uniform coordinates
        self.xind = np.random.randint(0, self._nx)
        self.yind = np.random.randint(0, self._ny)

        # Assume broken powerlaw source counts 
        nfluxhigh = np.random.uniform(0,1)**(-2./3.)
        nfluxlow = np.random.uniform(0,1)**(-1/1)
        self.flux = nfluxhigh*nfluxlow

#        if np.random.randint(0, 500) % 250 == 0:
#            self.flux = 10000.0
        
        # Galaxy size (semi-major axis)
        # From https://arxiv.org/abs/1601.03948
        self.sigx = 0.2 * np.random.gamma(2.25,1.) / self._pixel_size

        # Simulate ellipticity as Tunbridge et al. 2016
#        self.ellipticity=np.random.beta(1.7,4.5)
        self.ellipticity=np.random.beta(2*1.7,4.5)        # liam hack

        self.sigy = self.sigx * ((1-self.ellipticity)/(1+self.ellipticity))**0.5

        self.flux = self.flux #/ (self.sigx*self.sigy)

        self.coords = np.meshgrid(np.arange(0, self.nblock), np.arange(0, self.nblock))
        self.rho = np.random.uniform(-90,90)
        self.spec_ind = np.random.normal(0.55,0.25)

    def gaussian2D(self, 
                  coords=None,  # x and y coordinates for each image.
                  amplitude=1,  # Highest intensity in image.
                  xo=75,  # x-coordinate of peak centre.
                  yo=75,  # y-coordinate of peak centre.
                  sigma_x=3,  # Standard deviation in x.
                  sigma_y=3,  # Standard deviation in y.
                  rho=0,  # Correlation coefficient.
                  offset=0,
                  rot=0):  # rotation in degrees.
        if coords is None:
            self.galparams()
            coords = self.coords

        x, y = coords

        rot = np.deg2rad(rot)

        x_ = np.cos(rot)*x - y*np.sin(rot)
        y_ = np.sin(rot)*x + np.cos(rot)*y

        xo = float(xo)
        yo = float(yo)

        xo_ = np.cos(rot)*xo - yo*np.sin(rot) 
        yo_ = np.sin(rot)*xo + np.cos(rot)*yo

        x,y,xo,yo = x_,y_,xo_,yo_

        C = 4 * np.log(2)

        # Create covariance matrix
        mat_cov = [[C * sigma_x**2, rho * sigma_x * sigma_y],
                   [rho * sigma_x * sigma_y, C * sigma_y**2]]
        mat_cov = np.asarray(mat_cov)
        # Find its inverse
        mat_cov_inv = np.linalg.inv(mat_cov)

        # PB We stack the coordinates along the last axis
        mat_coords = np.stack((x - xo, y - yo), axis=-1)

        G = amplitude * np.exp(-np.matmul(np.matmul(mat_coords[:, :, np.newaxis, :],
                                                        mat_cov_inv),
                                              mat_coords[..., np.newaxis])) + offset
        return G.squeeze()

    def sis_alpha(self, x, y, theta_E):
        r = np.sqrt(x**2 + y**2)
        return theta_E * x / r, theta_E * y / r

    def lens_equation(self, beta_x, beta_y, alpha_x, alpha_y):
        return beta_x + alpha_x, beta_y + alpha_y

    def sersic2d(self, 
                 coords,  # x and y coordinates for each image.
                 amplitude=1,  # Highest intensity in image.
                 xo=75,  # x-coordinate of peak centre.
                 yo=75,  # y-coordinate of peak centre.
                 sigma_x=1,  # Standard deviation in x.
                 sigma_y=1,  # Standard deviation in y.
                 rho=0,  # Correlation coefficient.
                 ellipticity=0,
                 rot=0):  # rotation in degrees.
        mod = Sersic2D(amplitude=amplitude, r_eff=25, n=4, x_0=xo, y_0=yo, 
                       ellip=ellipticity, theta=np.deg2rad(rot))

        x,y = coords

        return mod(x,y)

    def distort_galaxy(self, 
                       gal_arr, 
                       alpha=20.0):
        gal_arr = gal_arr[:,:,None]*np.ones([1,1,3])
        gal_arr_distort = elastic_transform(gal_arr, alpha=alpha,
                                           sigma=3, alpha_affine=0)[:,:,0]
        return gal_arr_distort


    def get_coords(self, xind, yind, data):
        xmin, xmax = max(0,xind-self.nblock//2), min(xind+self.nblock//2,data.shape[0])
        ymin, ymax = max(0,yind-self.nblock//2), min(yind+self.nblock//2,data.shape[1])

        return xmin, xmax, ymin, ymax 

    def sim_sky(self, nsrc=None, noise=True, 
                background=False, fnblobout=None, 
                distort_gal=False):
        nchan = self._nchan
        nx, ny = self._nx, self._ny
        data = np.zeros([nx, ny, nchan])

        lensed_galaxies = []
        non_lensed_galaxies = []
        
        if nchan>1:
            freqarr = np.linspace(self._freqmin, self._freqmax, nchan)

        if nsrc is None:
            nsrc_ = self._src_density_sqdeg*(nx*ny*self._pixel_size**2/(3600.**2))
            nsrc = np.random.poisson(int(nsrc_))

        #print("Simulating %d sources" % nsrc)

        if background:
            pass

        for ii in range(nsrc):
            self.galparams()
            if fnblobout is not None:
                if ii==0:
                    self.write_gal_params(fnblobout, header=True)
                else:
                    self.write_gal_params(fnblobout, header=False)

            lens = np.random.randint(0, 10) == 5

            if lens:
                # Create simple Gaussian source
                n = 250  
                x = np.linspace(-2, 2, n)
                y = np.linspace(-2, 2, n)
                xx, yy = np.meshgrid(x, y)
                
                # Lens and source parameters
                theta_E = np.random.uniform(0.025, 0.3)
                beta_x, beta_y = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
                
                # Calculate deflection angles
                alpha_x, alpha_y = self.sis_alpha(xx, yy, theta_E)
                
                # Apply lens equation
                x_source, y_source = self.lens_equation(xx, yy, -alpha_x, -alpha_y)
    
                source_ii = self.flux * np.exp(-(x_source-beta_x)**2/(2*0.025**2) - \
                               (y_source-beta_y)**2/(2*0.02**2))
                
                lensed_galaxies.append({
                    'xind': self.xind,
                    'yind': self.yind,
                    'theta_E': theta_E,
                    'flux': self.flux,
                    'beta_x': beta_x,
                    'beta_y': beta_y
                })

            else:
                source_ii = self.gaussian2D(self.coords,
                                   amplitude=self.flux,
                                   xo=self.nblock//2,
                                   yo=self.nblock//2,
                                   sigma_x=self.sigx,
                                   sigma_y=self.sigy,
                                   rot=self.rho,
                                   offset=0)
                non_lensed_galaxies.append({
                    'xind': self.xind,
                    'yind': self.yind,
                    'flux': self.flux
                })

            if distort_gal is not False:
                alpha = distort_gal
                source_ii = self.distort_galaxy(source_ii, 
                                                alpha=alpha)

            xmin, xmax, ymin, ymax = self.get_coords(self.xind, self.yind, data)

            if nchan==1:
                data[xmin:xmax, ymin:ymax, 0] += (source_ii.T)[\
                            abs(min(0, self.xind-self.nblock//2)):min(self.nblock, self.nblock+nx-(self.xind+self.nblock//2)),\
                            abs(min(0, self.yind-self.nblock//2)):min(self.nblock, self.nblock+ny-(self.yind+self.nblock//2))]
            else:
                for nu in range(nchan):
                    spec_ind = self.spec_ind
                    Snu = (source_ii.T)[\
                                abs(min(0, self.xind-self.nblock//2)):min(self.nblock, self.nblock+nx-(self.xind+self.nblock//2)),\
                                abs(min(0, self.yind-self.nblock//2)):min(self.nblock, self.nblock+ny-(self.yind+self.nblock//2))]
                    Snu *= (freqarr[nu]/1.4)**(-spec_ind)
                    data[xmin:xmax, ymin:ymax, nu] += Snu

        return data, lensed_galaxies, non_lensed_galaxies

    def write_gal_params(self, fnout, header=False):
        f = open(fnout,'a+')

        if header:
            f.write('# xind  yind  sigx  sigy  orientation  flux\n')

        blobparams = (self.xind, self.yind, self.sigx, self.sigy, self.rho, self.flux)
        fmt_out = '%d  %d %0.2f %0.2f %0.3f %4f\n'
        f.write(fmt_out % blobparams)

    def write_data_fits(self, data, header, fnout):
        hdu = fits.PrimaryHDU(data, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(fnout)