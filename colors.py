from typing import Union
import re
import pygame
import numpy as np
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import numbers


colors = {**mpl_colors.CSS4_COLORS, **mpl_colors.TABLEAU_COLORS}
available_colormaps = ['Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']


def hex_to_RGB(hex_code: str) -> list:
    """
    Convert a hex color code into an RGB triplet

    Parameters
    ----------
    hex_code    A string specifying a hex color code

    Returns
    -------

    RGB         A 3-element RGB vector

    """

    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i + 2], 16) for i in (0, 2, 4)]


def is_valid_hex(hex_code: str) -> bool:
    """
    Determine whether a string is a valid hex color code

    Parameters
    ----------
    hex_code    A string

    Returns
    -------
    True is hex_code is a valid hex color code. Else False

    """

    if re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', hex_code):
        return True
    else:
        return False


def prepare_color(color: Union[str, tuple, np.ndarray], alpha=1) -> np.array:
    """
    Convert a color, which could be in any form, into a usuable RGB tuple

    Parameters
    ----------
    color       A color in any form

    Returns
    -------
    RGB         A (r, g, b, a) tuple
    """

    alpha = int(255 * alpha)

    if isinstance(color, str):

        # if it is a simple hex color code, convert to RGB and return
        if is_valid_hex(color):
            return np.array(hex_to_RGB(color) + [alpha]).astype(int)

        # if it is a preset matplotlib color, return the RGB code associated
        elif color in colors:
            return np.array(hex_to_RGB(colors[color]) + [alpha]).astype(int)

        else:
            raise ValueError('{} is not a valid color'.format(color))

    elif isinstance(color, tuple) or isinstance(color, list):

        # RGB tuple - add alpha
        if len(color) == 3:
            return np.array(list(color) + [alpha]).astype(int)

        # nothing to do
        elif len(color) == 4:
            return np.array(color).astype(int)

        else:
            raise ValueError('Invalid color tuple. Must be RGB(A) but has length {}'.format(len(color)))

    elif isinstance(color, np.ndarray):
        return color

    else:
        raise ValueError(f'color must be a string, tuple or pygame color to be passed to this function, but it is {type(color)}')


def color_mapper(cmap: str, vmin: float = 0, vmax: float = 1) -> callable:
    """
    This function returns a function that maps any number between vmin and vmax to a color, based on a chosen
    matplotlib color map

    Parameters
    ----------
    cmap        A valid matplotlib colormap string (like the cmap argument taken in plt.imshow)
    vmin        The minimum input value
    vmax        The maximum input value
    alpha       Possibility of adding alpha (between 0 and 1)

    Returns
    -------
    A function that takes a single value and outputs a color

    """

    cmap = plt.get_cmap(cmap)
    scalar_map = cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)

    def mapper(x: np.ndarray, alpha: np.ndarray = 1) -> np.ndarray:
        """
        This is the function that gets returned

        Parameters
        ----------
        x           A number between vmin and vmax, or a numpy array of numbers between vmin and vmax
        alpha       The desired alpha. Either a number between 0 and 1 or a numpy array
                    of the same length as x

        Returns
        -------
        An RGB color

        """

        if isinstance(x, np.ndarray) and isinstance(alpha, np.ndarray):

            if x.shape != alpha.shape:
                raise ValueError(
                    f"x and alpha should have the same shape but they are {x.shape} and {alpha.shape} respectively")

        if isinstance(x, np.ndarray):

            if isinstance(alpha, numbers.Number):
                return (255 * scalar_map.to_rgba(x, alpha=alpha)).astype(int)


            elif isinstance(alpha, np.ndarray):
                out = (255 * scalar_map.to_rgba(x, alpha=0)).astype(int)
                alpha[alpha < 0] = 0
                alpha[alpha > 1] = 1
                out[..., -1] = (255 * alpha).astype(int)
                return out

            else:
                raise ValueError(f"alpha should be a number or a numpy array, but it's {type(alpha)}")

        if isinstance(x, numbers.Number):

            if isinstance(alpha, np.ndarray):
                raise ValueError('x is a number, so alpha cannot be a numpy array')

            return (255 * np.array(scalar_map.to_rgba(x, alpha=alpha))).astype(int)

    return mapper


