#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:50:17 2021

@author: yaoling
"""

#%% Use packages
from skgstat import Variogram
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.random.seed(42)

print("hello world")

coordinates = np.random.gamma(20, 5, (50, 2))
values = np.random.normal(20, 5, 50)

V = Variogram(coordinates=coordinates, values = values)
print(V)
V.plot()


#%%

