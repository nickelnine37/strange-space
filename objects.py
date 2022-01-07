import pygame
import numpy as np
from colors import color_mapper, prepare_color
from transformations import rotate

class Object:

    def __init__(self, location: np.ndarray, position: str='absolute', plot_type='scatter'):
        self.location = location
        self.position = position
        self.plot_type = plot_type

    def get_points(self):
        pass


class Stars(Object):

    def __init__(self, N: int, size: int=3):
        super().__init__(location=np.zeros(3), position='relative', plot_type='scatter')

        stars = np.random.normal(size=(3, N))
        self.x, self.y, self.z = stars / (stars ** 2).sum(0) ** 0.5
        self.alpha = np.random.uniform(0, 255, size=N).astype(int)
        self.radii = 1 + (size * self.alpha / 255).astype(int)
        self.color = (255 * np.ones((N, 4))).astype(int)
        self.color[:, 3] = self.alpha

    def get_points(self):
        return self.x.copy(), self.y.copy(), self.z.copy(), self.radii.copy(), self.color.copy()

class LorenzAttractor(Object):

    def __init__(self,
                 location = np.zeros(3),
                 ititial_coords: tuple=(0.01, 0, 0),
                 scale: float = 1,
                 tail_length: float = 2000,
                 ball_radius_pixels: int=100,
                 color_map: str = 'hsv',
                 sigma: float = 10,
                 rho: float = 28,
                 beta: float = 8/3,
                 dt: float = 0.006):

        super().__init__(location=location, position='absolute', plot_type='scatter')

        self.scale = scale
        self.tail_length = tail_length
        self.ball_radius_pixels = ball_radius_pixels
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.x, self.y, self.z = ititial_coords
        self.points = np.zeros((tail_length, 3))
        self.i = 0

        self.color_mapper = color_mapper(color_map, vmin=0, vmax=1)

    def get_points(self):

        self.x += (self.sigma * (self.y -self. x)) * self.dt
        self.y += (self.x * (self.rho - self.z) - self.y) * self.dt
        self.z += (self.x * self.y - self.beta * self.z) * self.dt

        self.points[self.i % self.tail_length, :] = self.x, self.y, self.z
        self.i += 1

        return self.points[:self.i, 0], self.points[:self.i, 1], self.points[:self.i, 2], self.ball_radius_pixels, self.coords_to_cmap(self.i)

    def coords_to_cmap(self, t: int):
        return self.color_mapper(np.linspace(0, 1, self.points.shape[0]).take(range(5 * t, 5 * t + self.points.shape[0]), axis=0, mode='wrap'), alpha=0.8)



class StrangeAttractor1(Object):

    def __init__(self,
                 params: np.ndarray = None,
                 location: np.ndarray = np.zeros(3),
                 ititial_coords: tuple=(10, -5, 23),
                 scale: float = 1,
                 tail_length: float = 2000,
                 ball_radius_pixels: int=100,
                 color_map: str = 'hsv',
                 dt: float = 0.001):

        super().__init__(location=location, position='absolute', plot_type='scatter')

        self.scale = scale
        self.tail_length = tail_length
        self.ball_radius_pixels = ball_radius_pixels

        if params is None:
            self.a, self.b, self.c, self.d, self.e, self.f = np.random.uniform(-1, 1, size=6)
        else:
            self.a, self.b, self.c, self.d, self.e, self.f = params

        self.params = (self.a, self.b, self.c, self.d, self.e, self.f)
        self.dt = dt
        self.x, self.y, self.z = ititial_coords
        self.points = np.zeros((tail_length, 3))
        self.i = 0

        self.color_mapper = color_mapper(color_map, vmin=0, vmax=1)

    def get_points(self):

        self.x += (self.z * np.sin(self.a * self.x) + np.cos(self.b * self.y)) * self.dt
        self.y += (self.x * np.sin(self.c * self.y) + np.cos(self.d * self.z)) * self.dt
        self.z += (self.y * np.sin(self.e * self.z) + np.cos(self.f * self.x)) * self.dt

        # self.x = newx
        # self.y = newy
        # self.z = newz

        print( self.x, self.y, self.z)

        self.points[self.i % self.tail_length, :] = self.x, self.y, self.z
        self.i += 1

        return self.scale * self.points[:self.i, 0], self.scale * self.points[:self.i, 1], self.scale * self.points[:self.i, 2], self.ball_radius_pixels, self.coords_to_cmap(self.i)

    def coords_to_cmap(self, t: int):
        return self.color_mapper(np.linspace(0, 1, self.points.shape[0]).take(range(5 * t, 5 * t + self.points.shape[0]), axis=0, mode='wrap'), alpha=0.8)



class Circle(Object):

    def __init__(self, location: np.ndarray, normal: np.ndarray, radius: float=1):
        super().__init__(location, position='absolute', plot_type='ellipse')



class Square(Object):

    def __init__(self, location: np.ndarray=np.zeros(3),
                 rotation: np.ndarray=np.zeros(3),
                 scale: float=1,
                 color='white',
                 alpha: float=1):
        super().__init__(location=location, position='absolute', plot_type='polygon')

        self.coords = scale * rotate(*rotation) @ np.array([[0.5, 0.5, -0.5, -0.5],
                                                            [0.5, -0.5, -0.5, 0.5],
                                                            [0, 0, 0, 0]])


        self.color = color
        self.alpha = alpha

    def get_points(self):

        print(prepare_color(self.color, alpha=self.alpha))
        return self.coords[0, :].copy(), self.coords[1, :].copy(), self.coords[2, :].copy(), None, prepare_color(self.color, alpha=self.alpha)


class BitMap(Object):

    def __init__(self,
                 bitmap: np.ndarray,
                 location: np.ndarray=np.zeros(3),
                 rotation: np.ndarray=np.zeros(3),
                 scale: float=1):
        super().__init__(location=location, position='absolute', plot_type='bitmap')

        self.h, self.w = bitmap.shape
        rotation = rotate(*rotation)
        self.bitmap = bitmap

        def square(i, j):
            return np.array([[(i - self.w / 2), (i + 1 - self.w / 2), (i + 1 - self.w / 2), (i - self.w / 2)],
                             [(j - self.h / 2), (j - self.h / 2), (j + 1 - self.h / 2), (j + 1 - self.h / 2)],
                                [0, 0, 0, 0]])

        self.squares = [scale * rotation @ square(i, j) for i in range(self.w) for j in range(self.h)]

        print(self.squares[0])

    def get_points(self):
        return self.squares, self.bitmap.reshape(-1)






def ParamsToString(params):

    base27 = "_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    a = 0
    n = 0

    result = ''

    for i in range(18):

        a = a * 3 + int(params[i]) + 1
        n += 1

        if (n == 3):
            result += base27[a]
            a = 0
            n = 0

    return result


def StringToParams(string):

    params = [0] * 18
    string = string.upper()
    base27 = ' _ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for i in range(6):
        a = 0

        if i  < len(string):
            c = string[i]
        else:
            c = ' '

        if c >= 'A' or c <= 'Z':
            a = int(base27.index(c) - base27.index('A')) + 1


        params[i*3 + 2] = a % 3 - 1
        a /= 3
        params[i*3 + 1] = a % 3 - 1
        a /= 3
        params[i*3 + 0] = a % 3 - 1


    return params




if __name__ == '__main__':

    import time
    np.set_printoptions(precision=5, linewidth=500, threshold=500, suppress=True, edgeitems=5)

    S = StrangeAttractor1(location=np.random.uniform(-25, 25, size=3), dt=0.005)

    print(S.params)

    while True:
        S.get_points()
        time.sleep(0.1)
