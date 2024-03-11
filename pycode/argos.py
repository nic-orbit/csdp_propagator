import os
import sys
import inspect
import numpy as np
import urllib.request
from datetime import datetime

"""Accessing and Recording Global Organization System: a mediocre acronym to describe a module that fetches data like a loyal Argos.

The functions in this module save and load np.ndarrays, figures, TLE data and code.
"""

def nparray_saver(data_name: str, ndarray: np.ndarray[float]):
    """Saves a numpy array to a .npy file in the 'data' folder in project root.

    Arguments:
        data_name: name of the data that is being saved.
        ndarray: the numpy array to be saved.
    """

    # Create the path to the file to be saved
    code_path = os.path.dirname(__file__)
    main_path = os.path.dirname(code_path)
    file_path = os.path.join(main_path, 'data', data_name + '.npy')
    # Save the numpy array
    np.save(file_path, ndarray)
    return

def nparray_saver_npz(data_name: str, ndarray: np.ndarray[float]):
    """Saves a numpy array to a .npy file in the 'data' folder in project root.

    Arguments:
        data_name: name of the data that is being saved.
        ndarray: the numpy array to be saved.
    """

    # Create the path to the file to be saved
    code_path = os.path.dirname(__file__)
    main_path = os.path.dirname(code_path)
    file_path = os.path.join(main_path, 'data', data_name + '.npz')
    # Save the numpy array
    np.savez(file_path, ndarray)
    return

def fig_saver(fig_topic: str, fig):
    """Saves a matplotlib figure to a .png file in the 'plots' folder in project root.

    Arguments:
        fig_topic: the type of data that is being displayed.
        fig: the matplotlib figure to be saved.
    """

    # Create the path to the image to be saved
    code_path = os.path.dirname(__file__)
    main_path = os.path.dirname(code_path)
    file_path = os.path.join(main_path, 'plots', fig_topic + '.png')
    # Save the figure
    fig.savefig(file_path,dpi = 150)

    return


def norm_calc(array: np.ndarray[float]) -> np.ndarray[float]:
    """Calculates the norm of a vector along the first direction.
    
    Arguments:
        array: array to be elaborated.
    Returns:
        array_norm: array of the norms.
    """

    # Calculate the norm
    array_norm = np.sqrt(np.einsum('ij,ij->i', array, array))
    
    return array_norm


def angle_calc(array1: np.ndarray[float], array2: np.ndarray[float]) -> np.ndarray[float]:
    """Calculates the angle between two vectors or vector lists in radians.
    
    Arguments:
        array1: first array.
        array2: second array.
    Returns:
        angle: angle between the arrays in radians.
    """

    cos = np.einsum('ij,ij->i',array1,array2) * np.reciprocal(norm_calc(array1) * norm_calc(array2))
    cos = np.clip(cos, -1, 1)
    angle = np.arccos(cos)

    return angle


def evalPoly(lst: np.ndarray[float], x: np.ndarray[float]) -> np.ndarray[float]:
    """Function that evaluates a polynomial of a given degree over a range of values.
    
    Arguments:
        lst: list of the polynomial coefficients.
        x: array to evaluate the polynome over.
    Returns:
        res: resulting evaluated polynomial.
    """

    res = np.zeros(x.size)
    for power, coeff in enumerate(lst): # starts at 0 by default
        res += (x**power) * coeff
    return res