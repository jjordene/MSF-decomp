# Author: George Cherry <georgche@uio.no>


''' Most Significant Frequency decomposition by George Cherry, copyright June 2024

    Produces FFT in 1, 2 or 3 dimensions, relating to kt, kx and ky. Note: KT is always a real fft.
    Input: a 1-3 dimensional array
    Output: 3 arrays of same dimension as input array: 1) msf index, 2) Difference between, msf output and signal, 3) msf signal.
    
    Change log:
    
    29.09.2025 - Finished generalisation for 1-3 dimensions.
    27.10.2025 - Adapted to Pep-8 conventions.


'''


import numpy as np
import scipy as sci
from numba import njit, prange, get_num_threads, set_num_threads


@njit(parallel=True)
def filter_psd(psd, ks, bins, workers=-1):

    threads = get_num_threads() if workers == -1 else workers
    set_num_threads(threads)

    psd_bins = np.zeros((len(bins)+1,) + np.shape(psd), dtype='complex64')

    for k in prange(len(bins)):
        psd_bins[k] = np.where(np.abs(ks) == bins[k], psd, 0)

    psd_bins[-1] = np.where(np.abs(ks) > bins[-1], psd, 0)

    return psd_bins


@njit(parallel=True)
def filter_psd_kperp(psd, ks, bins, workers=-1):  # !!! Doesn't work - FIX
    threads = get_num_threads() if workers == -1 else workers
    set_num_threads(threads)

    psd_bins = np.zeros((len(bins)+1,)+np.shape(psd), dtype='complex64')
    for k in prange(len(bins)):
        psd_bins[k] = np.where((np.abs(ks) <= bins[k][0])
                               & (np.abs(ks) > [k][1]), psd, 0)

    psd_bins[-1] = np.where(np.abs(ks) > bins[-1][1], psd, 0)
    return psd_bins


@njit(parallel=True)
def find_diff(ifft_bins, dat,sec=False, workers=-1):
    threads = get_num_threads() if workers == -1 else workers
    set_num_threads(threads)

    diff = np.zeros(((len(ifft_bins),) + np.shape(dat)))
    for k in prange(len(ifft_bins)):
        if sec:
            diff[k] = np.abs(ifft_bins[k] - dat)
        else:
            diff[k] = np.abs(
                ifft_bins[k]-dat) if k == 0 else np.abs(ifft_bins[k]+ifft_bins[0] - dat)

    return diff



@njit(parallel=True)
def msf_par(diff, ifft_bins, sec=False, workers=-1):
    
    threads = get_num_threads() if workers == -1 else workers
    set_num_threads(threads)

    msf_ind = np.zeros(np.shape(diff[0]))
    msf_diff = np.zeros(np.shape(diff[0]))
    msf_wave = np.zeros(np.shape(diff[0]))

    nt, ny, nx = np.shape(diff[0, :, :, :])

    for i in prange(nt):
        for j in prange(ny):
            for k in prange(nx):
                index = np.argmin(diff[:, i, j, k])
                msf_ind[i, j, k] = index
                msf_diff[i, j, k] = min(diff[:, i, j, k])

                if sec or index == 0:
                    msf_wave[i, j, k] = ifft_bins[index][i, j, k]
                else:
                    msf_wave[i, j, k] = ifft_bins[index][i, j, k] + \
                        ifft_bins[0][i, j, k]

    return msf_ind, msf_diff, msf_wave


class msf:
    def __permute_dir2_dim3__(self,):
        y_ax = 3-(self.ax[0]+self.ax[1])
        freq_ax = self.ax + (y_ax,)
        b = np.array(freq_ax)
        a = (0, 1, 2)
        if np.any(b[b == np.array(a)]):
            trans_ax = freq_ax
            self.trans_r = freq_ax
        else:
            if freq_ax == (2, 0, 1):
                trans_ax = (1, 2, 0)
                self.trans_r = (2, 0, 1)
            else:
                trans_ax = (2, 0, 1)
                self.trans_r = (1, 2, 0)
        return trans_ax

    def __permute_dim3__(self, var):
        if (self.ax != (0, 1, 2)):
            b = np.array(self.ax)
            a = (0, 1, 2)

            if np.any(b[b == np.array(a)]):
                trans_ax = self.ax
                self.trans_r = self.ax
            else:
                if self.ax == (2, 0, 1):
                    trans_ax = (1, 2, 0)
                    self.trans_r = self.ax
                else:
                    trans_ax = (2, 0, 1)
                    self.trans_r = self.ax
            var = np.transpose(var, trans_ax)
        return var

    def __permute_data__(self, var):
        if (self.dims == 1) & (self.datdims != 1):  # 1 frequency dimension in ND data
            if (self.ax != 0):
                var = np.moveaxis(var, self.ax, 0)
                self.perm_r = self.ax
        elif (self.dims == 2):  # 2 frequency dimensions
            if (self.ax[:2] != (0, 1)):
                if self.datdims == 2:  # in 2D data
                    trans_ax = (1, 0)
                    self.trans_r = (1, 0)
                else:  # in 3D data
                    trans_ax = self.__permute_dir2_dim3__()
                var = np.transpose(var, trans_ax)
        elif (self.dims == 3):  # 3 frequency dimension (in 3D data)
            var = self.__permute_dim3__(var)

        return var

    def __init__(self, file, axes=(0, 1, 2), dt=None,dx=None,dy=None, normalize=False, normconst=0.5):

        var = file.copy()
        self.ax = axes
        self.datdims = np.ndim(var)
        # number of dimensions to be calculated
        self.dims = len(self.ax) if type(self.ax) is not int else 1

        self.trans_r = None
        self.perm_r = None
        # Transpose data to correct kt,kx,ky form

        var = self.__permute_data__(var)

        if normalize:
            self.normconst = normconst
            self.dat_non_normalized = var
            var = np.sign(var)*np.abs(var)**self.normconst
        self.dat = var
        
        self.dt,self.dx,self.dy=dt,dx,dy


    # Can be used to permute msf results or data to dimension original order
    def repermute_data(self, f):
        if self.perm_r is not None:
            out = np.moveaxis(f, 0, self.perm_r)
        elif self.trans_r is not None:
            out = np.transpose(f, self.trans_r)
        else:
            out = f
        return out

    def get_dims(self,):
        # Get size of file for each dimension
        if self.dims == 1:
            try:
                self.mt = np.shape(self.dat)[0]
            except ValueError:
                raise RuntimeError("File dimensions smaller than axis chosen")

        elif self.dims == 2:
            try:
                self.mt = np.shape(self.dat)[0]
                self.mx = np.shape(self.dat)[1]
            except ValueError:
                raise RuntimeError("File dimensions smaller than axes chosen")
        elif self.dims == 3:
            try:
                self.mt = np.shape(self.dat)[0]
                self.mx = np.shape(self.dat)[1]
                self.my = np.shape(self.dat)[2]
            except ValueError:
                raise RuntimeError("File dimensions smaller than axes chosen")

        else:
            raise RuntimeError(
                'Dimensions of selected axes must be in range 1-3.')

    def __get_axes_fft__(self,):
        if self.dims== 1:
            return None
        elif self.dims == 2:
            axes_fft = (1, 0)
        elif self.dims == 3:
            axes_fft = (1, 2, 0)

        return axes_fft

    def __get_axes_shift__(self,):
        if self.dims== 1:
            return None
        elif self.dims == 2:
            axes_shift = (1,)
        elif self.dims == 3:
            axes_shift = (1, 2)
        return axes_shift

    def __get_s__(self):
        if self.dims== 1:
            return None
        elif self.dims == 2:
            s = (self.mx, self.mt)
        elif self.dims == 3:
            s = (self.mx, self.my, self.mt)
        return s
    
        
    def __check_shape__(self,var):
        if np.shape(var) != np.shape(self.dat):
            var=self.__permute_data__(var)
            if np.shape(var)!=np.shape(self.dat):
                raise ValueError("Shape of data must be ", np.shape(self.dat))
        return var

    def psd(self, workers=-1): 
        if not hasattr(self, "mt"):
            self.get_dims()

        axes_fft = self.__get_axes_fft__()

        axes_shift = self.__get_axes_shift__()

        s = self.__get_s__()

        psd = sci.fft.rfft(self.dat, self.mt, axis=0, workers=workers) if self.dims == 1 else sci.fft.fftshift(
            sci.fft.rfftn((self.dat).astype(float), s=s, axes=axes_fft, workers=workers), axes=axes_shift)
        return psd

    def calc_freq(self,):
        # Calculate frequency range and resolution
        self.get_dims()

        if hasattr(self, "mx"):
            try:
                self.kx = sci.fft.fftshift(sci.fft.fftfreq(self.mx, self.dx))
                self.kx_pos = self.kx[self.kx >= 0]
            except TypeError:
                raise RuntimeError("Missing dx")

        if hasattr(self, "my"):
            try:
                self.ky = sci.fft.fftshift(sci.fft.fftfreq(self.my, self.dy))
                self.ky_pos = self.ky[self.ky >= 0]
            except TypeError:
                raise RuntimeError("Missing dy")

        if hasattr(self, "mt"):
            try:
                # Real fft on final dimension
                self.kt = sci.fft.rfftfreq(self.mt, self.dt)
            except TypeError:
                raise RuntimeError("Missing dt")

        # Create n-dimensional frequency grid

        if self.dims == 3:
            self.KTT, self.KXX, self.KYY = np.meshgrid(
                self.kt, self.kx, self.ky, indexing='ij')
        elif self.dims == 2:
            self.KTT, self.KXX = np.meshgrid(self.kt, self.kx, indexing='ij')
        else:
            self.KTT = self.kt

    def __get_k__(self, dir):
        if dir == 'kt':
            k = self.kt
        elif dir == 'kx':
            k = self.kx_pos

        elif dir == 'ky':
            k = self.ky_pos
        else:
            raise ValueError("Invalid frequency direction for bin.")
        return k

    def create_freqbins(self, dir='kt', f_max=-1):

        if not hasattr(self, "kt"):
            self.calc_freq()

        if dir == 'kperp':
            f_max = min(f_max, np.max(self.kx), np.max(self.ky)) if f_max >0 else min(
                np.max(self.kx), np, max(self.ky))
            target = self.kx_pos
            bounds = list(target[target <= f_max])
            bins = []
            bins.append([bounds[0], bounds[1]/2])
            for i in range(0, len(bounds)):
                if i >= 1:
                    bins.append([(target[i-1]+target[i])/2,
                                (target[i]+target[i+1])/2])
            self.bins = bins

        else:
            k = self.__get_k__(dir)

            f_max = min(f_max, np.max(k)) if f_max > 0 else np.max(k)
            self.bins = k[k <= f_max]

        return self.bins

    def __filter_psd_nopar__(self, psd, dir, bins, kperp=False):
        if self.datdims > 1:
            if dir == 'kt':
                ks = self.kt

            elif dir == 'kx':
                ks = self.KXX

            elif dir == 'ky':
                ks = self.KYY

            elif dir == 'kperp':
                ks = np.sqrt(self.KXX**2.+self.KYY**2.)
            else:
                raise ValueError("Invalid frequency direction.")
        else:
            ks = self.kt

        psd_bins = []
        if kperp:
            for i in range(0, len(bins)):
                psd_temp = np.copy(psd)
                psd_temp[ks <= bins[i][0]] = 0
                psd_temp[ks > bins[i][1]] = 0
                psd_bins.append(psd_temp)

            # Second case: all other values
            psd_temp = np.copy(psd)
            psd_temp[ks <= bins[-1][1]] = 0

            psd_bins.append(psd_temp)

        else:
            for i in range(len(bins)):
                psd_temp = np.copy(psd)
                psd_temp[np.abs(ks) != bins[i]] = 0

                psd_bins.append(psd_temp)

            # Second case: all other values
            psd_temp = np.copy(psd)
            psd_temp[np.abs(ks) < bins[-1]] = 0
            psd_bins.append(psd_temp)

        return psd_bins

    def __get_ks__(self, psd,dir):
        if dir == 'kt':

            if (self.datdims == 2) & (self.dims != 2):
                ks = np.meshgrid(self.kt, np.arange(
                    np.shape(psd)[1]), indexing='ij')[0]
            elif (self.datdims == 3) & (self.dims != 3):
                ks = np.meshgrid(self.kt, np.arange(np.shape(psd)[1]), np.arange(
                    np.shape(psd)[2]), indexing='ij')[0]
            else:
                ks = self.KTT

        elif dir == 'kx':
            if (self.datdims == 3) & (self.dims != 3):
                ks = np.meshgrid(np.arange(np.shape(psd)[0]), self.kx, np.arange(
                    np.shape(psd)[2]), indexing='ij')[1]
            else:
                ks = self.KXX

        elif dir == 'ky':
            ks = self.KYY
        else:
            raise ValueError("Invalid frequency direction.")

        return ks

    def __get_psd__(self, workers=-1,psd=None, **kwargs):
        if psd is None:
            print("No psd found. Calculating psd...")
            print(" ")
            psd = self.psd(workers=workers)
            print("psd calculated.")
            print(" ")
        return psd

    def __get_bins__(self, dir=None,f_max=None,bins=None, **kwargs):
        if bins is None:
            if (hasattr(self, 'bins')) & (f_max is None):
                bins = self.bins
            elif f_max is None:
                raise ValueError("f_max undefined")
            else:
                print("No frequency bins found. Creating bins...")
                print(" ")
                bins = self.create_freqbins(
                    dir=dir, f_max=f_max)

                print("Bins created.")
        return bins
    
    #kwargs: psd,bins,f_max
    def filtered_ifft(self, f_max=None, dir='kt', parallel=True, workers=-1, **kwargs):
        
        psd = self.__get_psd__(workers, **kwargs)
        
        bins = self.__get_bins__(dir,f_max, **kwargs)

        if parallel:
            if dir == 'kperp':
                ks = np.sqrt(self.KXX**2.+self.KYY**2.)
                print("Filtering psd...")
                print(" ")
                psd_bins = filter_psd_kperp(psd, ks, bins, workers)
                print("Done.")
            else:
                ks = self.kt if self.datdims == 1 else self.__get_ks__(psd,dir)
                print("Filtering psd...")
                print(" ")
                psd_bins = filter_psd(psd, ks, bins, workers)
                print("Done.")
        else:
            kp_log = (dir != 'kperp')
            psd_bins = self.__filter_psd_nopar__(psd, dir, bins, kperp=kp_log)

        s = self.__get_s__()
        axes_shift = self.__get_axes_shift__()
        axes_fft = self.__get_axes_fft__()

        ifft_bins = [sci.fft.irfft(psdD, axis=0, workers=workers)for psdD in psd_bins] if self.dims == 1 else [
            sci.fft.irfftn(sci.fft.ifftshift(psdD, axes=axes_shift), s=s, axes=axes_fft, workers=workers) for psdD in psd_bins]

        return ifft_bins

    # kwargs: psd, bins, parallel=True, workers=-1,dir=None,f_max=None
    def __get_ifft_bins__(self, ifft_bins=None, **kwargs):
        if ifft_bins is not None:
            ifft_bins = [self.__check_shape__(bin) for bin in ifft_bins]
            return ifft_bins
        else:
            print("No ifft_bins found. Calculating filtered fft bins...")
            print(" ")

            ifft_bins = self.filtered_ifft(**kwargs)
            print("Filtered fft bins calculated.")
            print(" ")
        return ifft_bins

    def point_diff(self, ifft_bins=None, sec=False, parallel=True, workers=-1, dir=None, **kwargs):  # kwargs: psd, bins
        if sec:
            try:
                dat = kwargs["data"]
                dat=self.__check_shape__(dat)
            except KeyError:
                raise ValueError("Must give new data on secondary iterations")
            if ifft_bins is None:
                raise ValueError("Must give new ifft_bins on secondary iterations")
        else:
            ifft_bins = self.__get_ifft_bins__(
                ifft_bins, parallel=parallel, workers=workers, dir=dir, **kwargs)

            dat = self.dat

        if parallel:
            diff_list = find_diff(ifft_bins=ifft_bins,
                                  dat=dat, sec=sec, workers=workers)
        else:
            diff_list = np.array([np.abs(ifft_bins[i]-dat) if i == 0 else np.abs(
                ifft_bins[i]+ifft_bins[0]-dat) for i in range(len(ifft_bins))])

        return diff_list

    def __get_point_diff__(self, ifft_bins, point_diff=None, **kwargs):  # kwargs: psd,bins
        if point_diff is not None:
            point_diff=[self.__check_shape__(diff) for diff in point_diff]
            return point_diff
        else:
            print("No point_diff found. Calculating point difference...")
            print(" ")
            point_diff = self.point_diff(ifft_bins, **kwargs)
            print("Point difference calculated.")
            print(" ")
        return point_diff 

    #kwargs: ifft_bins, data, point_diff, psd f_max=None
    def calc_msf(self, sec=False, parallel=False, workers=-1, dir='kt', f_max=-1,save_bins=False,  **kwargs):
        if sec:
            try:
                ifft_bins = kwargs["ifft_bins"]
            except KeyError:
                raise ValueError("Must give new ifft_bins on secondary iterations")
        else:    
            ifft_bins = self.__get_ifft_bins__(parallel=parallel, workers=workers,f_max=f_max,dir=dir, **kwargs)
            kwargs["ifft_bins"] = ifft_bins
        
        if save_bins:
            self.ifft_bins=ifft_bins
            
        point_diff = self.__get_point_diff__(sec=sec, parallel=parallel, workers=workers,**kwargs)

        print("Calculating msf...")
        print(" ")
        if parallel:
            
            if np.ndim(point_diff) == 2:
                point_diff = point_diff.reshape(np.shape(point_diff), 1, 1)
                ifft_bins = [i.reshape(np.shape(point_diff), 1, 1)
                             for i in ifft_bins]
            elif np.ndim(point_diff) == 3:
                point_diff = point_diff.reshape(np.shape(point_diff), 1)
                ifft_bins = [i.reshape(np.shape(point_diff), 1)
                             for i in ifft_bins]

            msf_index, msf_diff, msf_wave = msf_par(
                diff=point_diff, ifft_bins=ifft_bins, sec=sec, workers=workers)

        else:

            msf_index = np.argmin(point_diff, axis=0)

            print("Fetching difference...")
            print(" ")

            msf_diff = np.min(point_diff, axis=0)

            print("Creating msf wave...")
            print(" ")
            if sec:
                msf_wave = np.choose(msf_index, ifft_bins)
            else:
                msf_wave = np.choose(msf_index, ifft_bins) + \
                    np.where(msf_index != 0, ifft_bins[0], 0)
                    
        msf_wave=self.repermute_data(msf_wave)
        msf_index=self.repermute_data(msf_index)
        msf_diff=self.repermute_data(msf_diff)
        
        print("Finished")
        return msf_index, msf_diff, msf_wave

    def data_remainder(self,dat,msf_wave): 
        
        new_dat = dat - msf_wave
        return new_dat
    
    def bins_remainder(self, bins, ind):  #returns already permuted data for calc_msf
        if np.shape(bins[0]) != np.shape(ind):
            ind=self.__permute_data__(ind)
        try:
            bins_copy= [np.copy(bin) for bin in bins]
            for i in range(len(bins)):
                bins_copy[i][ind==i]=np.inf 
        except IndexError:
            raise RuntimeError("bins[i] and ind must be same shape")

        return bins_copy  

    
    