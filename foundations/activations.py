import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        sigma=[]
        for coord in z:
            sigma.append(round(1/(1+math.e**(-coord)),5))
            
        return np.array(sigma)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        vec=[]
        for coord in z:
            vec.append(round(max(0.0,coord),5))

        return np.array(vec)
