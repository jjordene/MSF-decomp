# 📦 MSF-decomp

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/jjordene/MSF-decomp?style=for-the-badge)](https://github.com/jjordene/MSF-decomp/stargazers)

[![GitHub forks](https://img.shields.io/github/forks/jjordene/MSF-decomp?style=for-the-badge)](https://github.com/jjordene/MSF-decomp/network)

[![GitHub issues](https://img.shields.io/github/issues/jjordene/MSF-decomp?style=for-the-badge)](https://github.com/jjordene/MSF-decomp/issues)

[![GitHub license](https://img.shields.io/github/license/jjordene/MSF-decomp?style=for-the-badge)](LICENSE)
<!-- TODO: If published to PyPI, add version and downloads badges:

[![PyPI version](https://img.shields.io/pypi/v/msf-decomp?style=for-the-badge)](https://pypi.org/project/msf-decomp/)

[![PyPI downloads](https://img.shields.io/pypi/dm/msf-decomp?style=for-the-badge)](https://pypi.org/project/msf-decomp/)
-->

**A Python module for the Most Significant Frequency (MSF) decomposition of signals.**

</div>

## 🎯 Why Choose MSF-decomp?

`MSF-decomp` provides an efficient and intuitive way to analyze and decompose complex signals by identifying and isolating its most prominent frequency components in space and time. This library is ideal for researchers, engineers, and data scientists working with signal processing, audio analysis, sensor data, and more, enabling deeper insights into the underlying periodicities and structures within their data.

## 🚀 Installation

`MSF-decomp` is a Python module. You can install it by cloning the repository and installing the necessary dependencies.

### Prerequisites
- Python 3.7+
- `numpy`
- `scipy`
- `numba` (for parallel functionalities)
- `matplotlib` (recommended for visualization features)

<!-- ### Install via pip (recommended)
If this package were to be published to PyPI, you would install it like this:
```bash
pip install msf-decomp
``` -->
<!-- TODO: Publish to PyPI and update this section -->

### Manual Installation (from source)
1. **Clone the repository**
   ```bash
   git clone https://github.com/jjordene/MSF-decomp.git
   cd MSF-decomp
   ```

2. **Install dependencies**
   ```bash
   pip install numpy scipy numba matplotlib
   ```

## 📖 Quick Start
There are two ways this module can be used. The easiest is to only pass the `calc_msf` function, and all stages are calculated at once. For large data this can take some time.
Here's a basic example of how to use `calc_msf` to decompose a synthetic signal and visualize the results.

```python
import numpy as np
import matplotlib.pyplot as plt
import MSF # Assuming MSF.py is in your Python path or current directory

# 1. Generate a synthetic signal with multiple frequencies
mx = 20
mt = 1024
my = 100
dt = dx = dy = 1
x,t,y = np.arange(mx),np.arange(mt),np.arange(my)
data=np.zeros((mx,mt,my))

kt = sci.fft.rfftfreq(mt,dt)

for yy in y:
    for tt in t:
        for xx in x:
            data[xx,tt,yy]= 0.5*np.cos(-2 * np.pi *kt[9]*tt*dt)+np.cos(-2 * np.pi *kt[2]*tt*dt)



# 2. Perform Most Significant Frequency (MSF) decomposition

# Decide order and number of MSF components
## We choose 3 components. The order of these components must be (mt,mx,my) therefore the order of the data (mx,mt,my) must change.
axes_order = (1,0,2)

#Initialise class
a = msf.msf(data,axes_order,dt,dy,dx)

#Calculate first iteration of MSF in the time direction (outputs: MSF frequency index, difference to original signal, MSF signal)
f_max = 0.02 #Maximum individual frequency calculated

msf_index,msf_diff,msf_wave = a.calc_msf(sec=False,parallel=True,f_max=f_max,save_bins=True,dir='kt')

#To iterate the MSF decomposition

##Calculate remainder of signal
new_sig = a.data_remainder(data,msf_wave)

## Calculate remainder of signal in each frequency bin
new_bins = a.bins_remainder(a.ifft_bins,msf_index)

#Iterate MSF calculation

msf2_index,msf2_diff,msf2_wave = a.calc_msf(data=new_sig,ifft_bins=new_bins,sec=True,parallel=True,save_bins=False)



# 3. Visualise the significant frequencies
fig,ax=plt.subplots(1,2)

ax[0].scatter(t,msf_index[0,:,0],s=1,color='lightblue',label="first_it")
ax[0].scatter(t,msf2_index[0,:,0],s=1,color='k',label="second_it")

ax[0].set_yticks(np.arange(0,len(a.kt[a.kt<f_max])+1),labels= [f"{a.kt[j]*1000:.2f}"  if a.kt[j] < f_max else f"$>$ {a.kt[j]*1000:.2f}" for j in range(len(a.kt[a.kt<f_max])+1)])

ax[0].set_ylabel("MSF (mHz)")
ax[0].set_xlabel("t")
ax[0].legend()


# Visualise the MSF signal against the original
ax[1].plot(t, data[0,:,0], label='Original Signal', color='gray', alpha=0.7)
ax[1].plot(t, msf_wave[0,:,0], label='First_it', color='lightblue', alpha=1)
ax[1].plot(t, msf2_wave[0,:,0]+msf_wave[0,:,0], label='Second_it', color='lightcoral', alpha=1)
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('t')
ax[1].legend()

```

It is also possible to do the MSF algorithm step by step, as follows:
- `psd` - calculates PSD through `scipy.fft`
- `filtered_ifft` - frequency filtering of the PSD, returning signals for each induvidual frequency
- `point_diff` - Calculates the difference between each filtered frequency signal and the original signal at each grid point.
- `calc_msf` - Returns the MSF at each point, the point diff for the MSF signal, and the MSF signal.

## 📚 API Documentation

The core functionality of `MSF.py` is exposed through the `MSF` class, primarily through the `calc_msf` function.

### msf.msf class
#### class `msf(file,axes=(0,1,2), dt=None, dx=None, dy=None, normalize=False, normconst=0.5)`
**Parameters**

- `file` (`np.ndarray`): \
Data for the MSF decomposition to be performed on. Can be 1, 2 or 3 dimensional.

- `axes` (`tuple` or `int`): \
Dimensions for Fourier transform to be performed on. The first dimension will always relate to `self.kt`, and will use `scipy.fft.rfft`. Therefore, the number of outputs will be half the size of the original data in this direction. The second dimension relates to `self.kx` and the final to `self.ky`. The parameter `dy` is therefore only needed if the fft is performed in 3 dimensions, and similarly, `dx` for two or more dimensions.

- `dt`, (`float`): \
Step size of first dimension, $t$.

- `dx`,`dy` (`float`, optional): \
Step sizes for second and third dimensions. Only needed if `axes` is a tuple of length 2 or 3, respectively.


### Core Functions

#### `calc_msf(self, sec=False, parallel=False, workers=-1, dir=None,save_bins=False, **kwargs)`
Performs Most Significant Frequency (MSF) decomposition on a given signal in the direction `dir`. In the case `dir='kperp'`, will calculate MSF bins which contain any `kx` and `ky` value between `kx**2+ky**2` steps.

**Parameters:**

- `sec` (`logical`): \
Treats as a first iteration if `False`. If `True`, requires the `data` and `ifft_bins` parameters, recalculated after first iteration.

- `parallel` (`logical`):\
 If True, uses parallelised version of MSF calculation. NOTE: Does not apply to the fft functionality. If no parallelisation is wanted in the fourier transforms, must set `workers` to `None`.
 
- `workers` (`int`, optional):\
 Maximum number of workers to use for parallel computation in fft. See `scipy.fft` documentation for more details.

- `dir` (`str`,optional):\
 Direction in which MSF is to be calculated. Possible directions: `'kt'`,`'kx'`,`'ky'` and `'kperp'`, depending on dimensions of `axes`. By default will calculate the temporal frequency (`kt`). Not needed if `ifft_bins` is given.

- `f_max` (`float`,optional):\
 Maximum individual frequency which will be calculated. By default all frequencies are treated individually (which can be computationally expensive).

- `save_bins` (`logical`, optional): \
If true, saves the attribute `ifft_bins`. This is needed for the second iteration of `calc_msf`.

- `ifft_bins` (`sequence of numpy.ndarrays`, optional):\
 List filtered bins from the inverse fourier transform for each frequency. If passed, `calc_msf` will skip the filtering process.

- `data` (`numpy.ndarray`,optional): \
Raw data for filtered signals to be compared to. Only necessary in additional iterations.

- `point_diff` (`sequence of numpy.ndarrays`, optional):\
Sequence of arrays containing the difference between each filtered frequency signal and the original data at each grid point. If passed, `calc_msf` will skip the difference calculation.

- `psd` (`numpy.ndarray`, optional):\
Normal DFT PSD of the data using `scipy` functionality. If passed, `calc_msf` will skip calculating the FFT.



**Returns:**
- `msf_index` (`numpy.ndarray`):\
The MSF index, relating to the index of the frequency array `k` which correlates to the MSF at each grid point.
- `msf_diff` (`numpy.ndarray`): \
The difference between the MSF signal, `msf_wave` and the original signal.
- `msf_wave` (`numpy.ndarray`):\
The signal created by the MSF at each grid point.

**Raises**
- `ValueError` \
If `ifft_bins` or `data` are not given when `sec=True`


**Example:**
```python
import numpy as np
import MSF


# 1. Generate a synthetic signal with multiple frequencies
mx = 20
mt = 1024
my = 100
dt = dx = dy = 1
x,t,y = np.arange(mx),np.arange(mt),np.arange(my)
data=np.zeros((mx,mt,my))

kt = sci.fft.rfftfreq(mt,dt)

for yy in y:
    for tt in t:
        for xx in x:
            data[xx,tt,yy]= 0.5*np.cos(-2 * np.pi *kt[9]*tt*dt)+np.cos(-2 * np.pi *kt[2]*tt*dt)



# 2. Perform Most Significant Frequency (MSF) decomposition
#Initialise class
a = msf.msf(data,(1,0,2),dt,dy,dx)

#Calculate first iteration of MSF in the kt direction 
f_max = 0.02 

msf_index,msf_diff,msf_wave = a.calc_msf(sec=False,parallel=True,f_max=f_max,save_bins=True,dir='kt')

#msf_index gives the index of kt which relates to the MSF at each point.
```

## 🛠️ Tech Stack

-   **Runtime:** ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
-   **Libraries:**
    -   ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) (Numerical operations)
    -   ![SciPy](https://img.shields.io/badge/scipy-%23EB2E2E.svg?style=for-the-badge&logo=scipy&logoColor=white) (Scientific computing, signal processing)
    -   ![Numba](https://img.shields.io/badge/Numba-3670A0?style=for-the-badge&logo=numba&logoColor=white) (Parallelisation)

    -   ![Matplotlib](https://img.shields.io/badge/Matplotlib-white?style=for-the-badge&logo=matplotlib&logoColor=black) (Plotting and visualization)

## 📁 Project Structure

```
MSF-decomp/
├── .gitignore          # Specifies intentionally untracked files to ignore
├── LICENSE             # Project's license (BSD 3-Clause)
├── MSF.py              # The core MSF decomposition module
└── README.md           # This documentation file
```

## 🤝 Contributing

We welcome contributions to `MSF-decomp`! If you'd like to contribute, please consider the following:

### Development Setup
1.  Fork the repository.
2.  Clone your forked repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/MSF-decomp.git
    cd MSF-decomp
    ```
3.  Install development dependencies (same as runtime dependencies for this project):
    ```bash
    pip install numpy scipy matplotlib
    ```
4.  Make your changes, add new features, or fix bugs.
5.  Ensure your code adheres to Python best practices (e.g., PEP 8).

### Running Tests
As of now, explicit test files are not present. Contributions including unit tests using a framework like `pytest` would be highly appreciated.

### Submitting Changes
1.  Commit your changes with clear and concise commit messages.
2.  Push your changes to your fork.
3.  Open a pull request to the `main` branch of the original `MSF-decomp` repository.

## 📄 License

This project is licensed under the [BSD 3-Clause "New" or "Revised" License](LICENSE) - see the [LICENSE](LICENSE) file for details.


## 📞 Support & Contact

-   🐛 Issues: [GitHub Issues](https://github.com/jjordene/MSF-decomp/issues) - For bug reports, feature requests, or questions.

---

<div align="center">

**⭐ Star this repo if you find it helpful for your signal processing needs!**

Made by George Cherry (jjordene)

</div>