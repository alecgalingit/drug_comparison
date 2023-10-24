from scipy.stats import zscore
import numpy as np

test = np.array([[2., 4.], [6., 8.]])
# print(test, '\n')
# print(zscore(test))

a = np.array([0.7972,  0.0767,  0.4383,  0.7866,  0.8091,
             0.1954,  0.6307,  0.6599,  0.1065,  0.0508])

print(zscore(a))
