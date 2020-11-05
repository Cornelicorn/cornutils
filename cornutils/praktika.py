import numpy as np
from typing import Callable, Tuple
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt

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
         connect: str = '', # e.g. 'dotted'
         label: Tuple[str, str, str, str, str, str] = ('x', 'y','Daten','Fit', 'Fit + $\sigma$', 'Fit - $\sigma$'),
        ):
    '''
    Plots `(x, y)` with optional errors `(sx, sy)` and the
    fit given a function `func` and a Tuple `regression_erg`
    which consists of the `beta` array and the error in `beta`
    '''
    fig = plt.figure(figsize = (10,6))
    
    plt.errorbar(x, y, fmt='rx',
                 label=label[2],
                 xerr=sx, yerr=sy, ecolor='black',
                 dash_capstyle='butt', capsize=3,
                 ls = connect)

    xmin = np.amin(x)
    xmax = np.amax(x)
    t = np.arange(xmin, xmax, (xmax-xmin)/2000)
    
    plt.plot(t, func(regression_erg[0],t), label = label[3])
    plt.plot(t, func(regression_erg[0]+regression_erg[1],t), label = label[4])
    plt.plot(t, func(regression_erg[0]-regression_erg[1],t), label = label[5])
    
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.legend(loc='lower right')
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
