# cornutils
Python package containing utility functions

## Install
```bash
python3 -m pip install cornutils
```

## Example Usage of praktika
```python
import numpy as np
from cornutils.praktika import Data, PlotSettings, aio

x = np.array([0,2,4,6,8,10,12,14,16,18,])
y = np.array([2.8,5.2,6.8,9.6,11.2,14,15.6,17,19.8,21.4,])
delta_y = np.array([1.0,1.5,1.2,0.9,1.4,1.4,1.5,1.2,1.3,1.5,])
data = Data(x, y, sy=delta_y)

s = PlotSettings(label_x='x-label', label_y = 'y-label',label_fit='Test')

def func(beta, x):
    '''
    Function to use for estimation, beta is an array with each parameter
    '''
    y = beta[0] + beta[1] * x
    return y

aio(func, 2, data, s)
```