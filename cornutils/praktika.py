import numpy as np
from typing import Callable, Tuple, Any
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
    title: str, optional
        Title for the plot. Defaults to None
    file: str, optional
        Path for a file to save the figure to. Defaults to None.
    dpi: int, optional
        Only useful with file specififed, dpi for the saved file.
        Defaults to 'figure', which will use the figures DPI
    file_format: str, optional
        Only useful with file specififed, file format of the saved plot.
        Defaults to 'png'.
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
    title: str = None
    file: str = None
    dpi: Any = 'figure'
    file_format: str = 'png'


@dataclass
class Data:
    """Dataclass for plot/regression data

    Attributes
    ----------
    x: np.ndarray
        x values
    y: np.ndarray
        y values
    sx: np.ndarray, optional
        Standard deviation in x. Defaults to None
    sy: np.ndarray, optional
        Standard deviation in y. Defaults to None
    """
    x: np.ndarray
    y: np.ndarray
    sx: np.ndarray = None
    sy: np.ndarray = None

def find_nearest_idx(array: np.ndarray,
                value
                ):
    """Find nearest index to value in array"""
    return np.argmin( np.abs(array-value) )

def estimation(x: np.ndarray,
               y: np.ndarray,
               num_params: int,
              ):
    "Stub algorithm to do beta0 estimation"
    beta = np.ones(num_params)
    beta[0] = y[find_nearest_idx(x, 0)]
    return beta

def regression(func: Callable,
               num_params: int,
               data: Data,
               estimation: Callable = estimation,
              ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Do an ODR
    Computes the Regression of func, given values
    and optional standard deviations and a function
    model.

    Parameters
    ----------
    func: Callable
        Function to use for estimation.
        Must be func(beta ,x) -> y.
    num_params: int
        Number of parameters used in func. Length for beta0.
    data: Data
        Regression data.

    Returns
    -------
    beta, sbeta, chi2_red, r2: Tuple[np.ndarray, np.ndarray, float, float]
        beta is the optimal valeu for beta in func, sbeta the
        standard deviation. chi2_red is the reduced $\chi^2$
        for the regression, r2 is $R^2$ for the regression.
    """
    realdata = RealData(data.x, data.y, sx=data.sx, sy=data.sy)
    model = Model(func)
    odr = ODR(realdata, model, beta0 = estimation(data.x,data.y, num_params))

    # If sx is given do odr, else just do least-squares
    fit_type = 0 if isinstance(data.sx, np.ndarray) else 2
    odr.set_job(fit_type=fit_type)

    output = odr.run()

    chi2_red = 0
    r2 = 0
    if isinstance(data.sy, np.ndarray):
        residuals = data.y - func(output.beta, data.x)
        chi_arr =  residuals / data.sy
        chi2_red = np.sum(chi_arr**2) / (len(data.x)-len(output.beta))
        ybar = np.sum(data.y/(data.sy**2))/np.sum(1/data.sy**2)
        r2 = 1 - np.sum(chi_arr**2)/np.sum(((data.y-ybar)/data.sy)**2)

    return output.beta, output.sd_beta*np.sqrt(len(data.x)), chi2_red, r2

def plotData(data: np.ndarray):
    data = np.asarray(data)

    fig = plt.figure(figsize = (10,6))
    for obj in data:
        plt.errorbar(obj.x, obj.y,
                 xerr=obj.sx, yerr=obj.sy,
                 dash_capstyle='butt', capsize=3,)


def plot(regression_erg: Tuple[np.ndarray, np.ndarray],
         func: Callable,
         data: Data,
         s: PlotSettings = PlotSettings(),
         newfig: bool = True,
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
    s: PlotSettings, optional
        PlotSettings object including labels and other options.
    """
    if newfig:
        fig = plt.figure(figsize = (10,6))

    plt.errorbar(data.x, data.y, fmt='rx',
                 label=s.label_data,
                 xerr=data.sx, yerr=data.sy, ecolor='black',
                 dash_capstyle='butt', capsize=3,
                 ls = s.connect)

    xmin = np.amin(data.x)
    xmax = np.amax(data.x)
    t = np.arange(xmin, xmax, (xmax-xmin)/2000)

    plt.plot(t, func(regression_erg[0],t), label = s.label_fit)
    if s.plot_sigp:
        plt.plot(t, func(regression_erg[0]+regression_erg[1],t), label = s.label_fit_sigp)
    if s.plot_sigm:
        plt.plot(t, func(regression_erg[0]-regression_erg[1],t), label = s.label_fit_sigm)

    plt.xlabel(s.label_x)
    plt.ylabel(s.label_y)
    plt.title(s.title)
    plt.legend(loc=s.loc_legend)
    if s.file:
        plt.savefig(f"{s.file}.{s.file_format}", dpi=s.dpi, format=s.file_format)

def aio(func: Callable,
        num_params: int,
        data: Data,
        estimation: Callable = estimation,
        return_result: bool = False,
        **kwargs,
       ):
    """Do an ODR and plot it
    Runs regression and plots the result

    Parameters
    ----------
    func: Callable
        Function to use for estimation.
        Must be func(beta ,x) -> y.
    num_params: int
        Number of parameters used in func. Length for beta0.
    data: Data
        Regression data.
    s: PlotSettings, optional
        PlotSettings object including labels and other options.

    Returns
    -------
    Prints out optimal parameters, respective standard
    deviation as well as the reduced $\chi^2$ and $R^2$.
    Returns plot of the result.
    """
    regression_erg = regression(func, num_params, data, estimation)
    print(f'''
    beta: \t {regression_erg[0]}
    sbeta: \t {regression_erg[1]}
    chi2: \t {regression_erg[2]}
    r2: \t {regression_erg[3]}
    '''
    )
    plot(regression_erg, func, data, **kwargs)
    if return_result:
        return regression_erg
