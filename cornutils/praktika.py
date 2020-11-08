import numpy as np
from typing import Callable, Tuple
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PlotSettings:
    """Plot settings
    Plot settings dataclass

    Attributes
    ----------
    label_x: str, optional
        Label for the x axis. Defaults to 'x'.
    label_y: str, optional
        Label for the x axis. Defaults to 'y'.
    label_data: str, optional
        Label for the errorbar plot. Defaults to 'Daten'.
    label_fit: str, optional
        Label for the Fit plot. Defaults to 'Fit'
    label_fit_sigp: str, optional
        Label for the Fit plot with added standard deviation. Defaults to 'Fit + $\sigma$'.
    label_fit_sigm: str, optional
        Label for the Fit plot with substracted standard deviation. Defaults to 'Fit - $\sigma$'.
    plot_sigp: bool, optional
        Whether to plot the $+\sigma$ function. Default to true
    plot_sigm: bool, optional
        Whether to plot the $-\sigma$ function. Defaults to true
    connect: str, optional
        Connection method for the data points, e.g. 'dotted'. Defaults to None.
    loc_legend: str, optional
        Location of the legend. Defaults to 'lower right'
    """
    label_x: str = 'x'
    label_y: str = 'y'
    label_data: str = 'Daten'
    label_fit: str = 'Fit'
    label_fit_sigp: str = 'Fit + $\sigma$'
    label_fit_sigm: str = 'Fit - $\sigma$'
    plot_sigp: bool = True
    plot_sigm: bool = True
    connect: str = None
    loc_legend: str = 'lower right'


def find_nearest_idx(array: np.ndarray,
                value
                ):
    return np.argmin( np.abs(array-value) )

def estimation(x: np.ndarray,
               y: np.ndarray,
               num_params: int,
              ):
    beta = np.ones(num_params)
    beta[0] = y[find_nearest_idx(x, 0)]
    return beta

def regression(func: Callable,
               num_params: int,
               x: np.ndarray,
               y: np.ndarray,
               sx: np.ndarray = None,
               sy: np.ndarray = None,
              ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    '''
    Computes the Regression of func, given values `x` and `y`
    and optional errors `sx` and `sy`. `func` should be
    `func(beta, x) -> y` and beta an array of length
    num_params. Returns `beta`, deviation of `beta` as well
    as reduced $\chi^2$ and $\R^2$
    '''
    data = RealData(x, y, sx=sx, sy=sy)
    model = Model(func)
    odr = ODR(data, model, beta0 = estimation(x,y, num_params))
    
    # If sx is given do odr, else just do least-squares
    fit_type = 0 if isinstance(sx, np.ndarray) else 2
    odr.set_job(fit_type=fit_type)
    
    output = odr.run()
    
    residuals = y - func(output.beta, x)
    chi_arr =  residuals / sy
    chi2_red = np.sum(chi_arr**2) / (len(x)-len(output.beta))
    ybar = np.sum(y/(sy**2))/np.sum(1/sy**2)
    r2 = 1 - np.sum(chi_arr**2)/np.sum(((y-ybar)/sy)**2)
    
    return output.beta, output.sd_beta*np.sqrt(len(x)), chi2_red, r2


def plot(regression_erg: Tuple[np.ndarray, np.ndarray],
         func: Callable,
         x: np.ndarray,
         y: np.ndarray,
         sx: np.ndarray = None,
         sy: np.ndarray = None,
         s: PlotSettings = PlotSettings(),
        ):
    """Quickly plot data and ODR regression
    Plots experimental data with optional standard deviations
    as well as ODR regression.

    Parameters
    ----------
    regression_erg: Tuple[np.ndarray, np.ndarray]
        Output of regression.
    func: Callable
        Callable to use for plotting the data.
    x: np.ndarray
        x values.
    y: np.ndarray
        y values.
    sx: np.ndarray
        Standard deviations in x. Defaults to None
    sy: np.ndarray
        Standard deviations in y. Defaults to None
    s: PlotSettings
        PlotSettings object including labels and other options.
    """
    fig = plt.figure(figsize = (10,6))
    
    plt.errorbar(x, y, fmt='rx',
                 label=s.label_data,
                 xerr=sx, yerr=sy, ecolor='black',
                 dash_capstyle='butt', capsize=3,
                 ls = s.connect)

    xmin = np.amin(x)
    xmax = np.amax(x)
    t = np.arange(xmin, xmax, (xmax-xmin)/2000)
    
    plt.plot(t, func(regression_erg[0],t), label = s.label_fit)
    if s.plot_sigp:
        plt.plot(t, func(regression_erg[0]+regression_erg[1],t), label = s.label_fit_sigp)
    if s.plot_sigm:
        plt.plot(t, func(regression_erg[0]-regression_erg[1],t), label = s.label_fit_sigm)
    
    plt.xlabel(s.label_x)
    plt.ylabel(s.label_y)
    plt.legend(loc=s.loc_legend)
    plt.show()

def aio(func: Callable,
        num_params: int,
        x: np.ndarray,
        y: np.ndarray,
        sx: np.ndarray = None,
        sy: np.ndarray = None,
       ):
    '''
    Computes ODR/LS and prints output of regression 
    '''
    regression_erg = regression(func, num_params, x, y, sx=sx, sy=sy)
    print(f'''
    beta: \t {regression_erg[0]}
    sbeta: \t {regression_erg[1]}
    chi2: \t {regression_erg[2]}
    r2: \t {regression_erg[3]}
    '''
    )
    plot(regression_erg, func, x, y ,sx=sx, sy=sy)
