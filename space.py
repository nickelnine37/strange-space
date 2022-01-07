import time

import numpy as np
import sys
from colors import *
import pygame.gfxdraw
from transformations import rotate_x, rotate_y, rotate_z, rotate, norm
from objects import *


class Space:
    """
    Screen class. This
    """

    def __init__(self,
                 screen_pixels: tuple = (1920, 1080),
                 screen_coords: tuple = (5, 2.8125),
                 camera_initial_position: tuple = (0, 0, -5),
                 viewing_plane_distance: float = 2,
                 add_stars: bool = True,
                 frame_rate: int = 30):
        """
        Initialise a space object.

        Parameters
        ----------
        screen_pixels               (Px, Py) the number of pixels in each direction for the pygame screen
        screen_coords               (X, Y) the equivelent width and height of the viewing plane in real space
        camera_initial_position     (cx, cy, cz) the inital position of the camera
        camera_initial_orientation  (theta_x, theta_y, theta_z) the Tait–Bryan angles of the camera orientation, measured
                                    in degrees. (0, 0, 0) points squarely to original viewing plane
        viewing_plane_distance      The distance (>0) in real space between the camera and the viewing plane (in the
                                    z direction)
        add_stars                   Boolean value: whether to add a starry background
        frame_rate                  Animation frame rate. Set to None to go as fast as possible

        """

        # the x-y coordinates of the viewing plane
        self.screen_coords = screen_coords
        self.x0, self.x1 = -screen_coords[0] / 2, screen_coords[0] / 2
        self.y0, self.y1 = -screen_coords[1] / 2, screen_coords[1] / 2

        # the coordinate system of the pixels
        self.screen_pixels = screen_pixels
        self.Px, self.Py = screen_pixels
        self.canvas = pygame.display.set_mode(screen_pixels)

        # convert between the two
        self.scale_pixels_to_coords = {'x': screen_coords[0] / screen_pixels[0],
                                       'y': screen_coords[1] / screen_pixels[1]}
        self.scale_coords_to_pixels = {'x': screen_pixels[0] / screen_coords[0],
                                       'y': screen_pixels[1] / screen_coords[1]}

        # these can potentially change
        self.camera_position = np.array(camera_initial_position).astype(float)
        self.camera_angle = np.zeros(3).astype(float)
        self.initial_direction = np.array([0, 0, 1]).astype(float)

        assert viewing_plane_distance > 0
        self.distance = viewing_plane_distance
        self.rotation_matrix = np.eye(3).astype(float)

        # fill in the canvas background
        self.background_color = prepare_color('black')
        self.canvas.fill(self.background_color)

        # should stay constant
        self.FRAME_RATE = frame_rate
        self.clock = pygame.time.Clock()

        # initialise velocity
        self.velocity = np.zeros(3).astype(float)

        # add pilot guide lines
        # self.pilot_lines = pygame.image.load('assets/images/pilot.png')

        if add_stars:
            self.objects = [Stars(500)]
        else:
            self.objects = []

        self.spacecraft = SpaceCraft(location=np.array([0, 0, 100]))



    def coords_to_pixels(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Convert a position in the x-y viewing plane to a pixel coordinate.

        Note that no check is performed to see whether the resulting pixels acutally fall within the range
        of the screen: this must be done separately.

        Parameters
        ----------
        x          The x-position in the x-y plane
        y          The y-position in the x-y plane

        Returns
        -------
        px         The pixel x-coordinate
        py         The pixel y-coordinate
        """
        # type checks
        if type(x) != type(y):
            raise ValueError(f'px and py must be of the same type but they are {type(x)} and {type(y)}')

        # numpy array checks
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.shape != y.shape:
                raise ValueError(f'x and y must have the same shape, but theyhave shapes {x.shape} and {y.shape} respectively')

        # get the pixels associated with coordinates (x, y) in the viewing plane
        px = (x + self.x1) * self.scale_coords_to_pixels['x']
        py = (self.y1 - y) * self.scale_coords_to_pixels['y']

        return px, py


    def pixels_to_coords(self, px: np.ndarray, py: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Convert a position in pixels on the screen, to an x-y coordinate on the viewing plane

        Parameters
        ----------
        px         The pixel x-coordinate
        py         The pixel y-coordinate

        Returns
        -------
        x          The x-position in the x-y plane
        y          The y-position in the x-y plane
        """

        # type checks
        if type(px) != type(py):
            raise ValueError(f'px and py must be of the same type but they are {type(px)} and {type(py)}')

        # numpy array checks
        if isinstance(px, np.ndarray) and isinstance(py, np.ndarray):
            if px.shape != py.shape:
                raise ValueError(f'px and py must have the same shape, but theyhave shapes {px.shape} and {py.shape} respectively')

        x = self.scale_pixels_to_coords['x'] * px - self.x1
        y = self.y1 - self.scale_pixels_to_coords['y'] * py

        return x, y


    def project_to_viewing_plane(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Project a point (or multiple points) existing at coordinates (x, y, z) onto the viewing plane

        Parameters
        ----------
        x          The x-coordinate(s) in real space
        y          The y-coordinate(s) in real space
        z          The z-coordinate(s) in real space

        Returns
        -------
        x'         The x-coordinate(s) in the viewing plane
        y'         The y-coordinate(s) in the viewing plane

        """


        transf_x, transf_y, transf_z = self.spacecraft.rotation_matrix @ (np.array([x, y, z]).reshape(3, -1) - self.spacecraft.location.reshape(3, 1))

        if all(isinstance(u, numbers.Number) for u in (x, y, z)):

            if transf_z <= 0:
                return np.nan
            else:
                return transf_x * self.distance / transf_z, transf_y * self.distance / transf_z

        elif all(isinstance(u, np.ndarray) for u in (x, y, z)):

            assert x.shape == y.shape
            transf_z[transf_z <= 0] = np.nan

            return transf_x * self.distance / transf_z, transf_y * self.distance / transf_z

        else:
            raise ValueError(f'x, y and z must be of the same type and either numbers or numpy arrays but they are {type(x)}, {type(y)} and {type(z)}')


    def scatter(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, radius: np.ndarray=500, color: np.ndarray='white'):
        global Tt

        assert all([isinstance(i, np.ndarray) for i in [x, y, z]])
        assert x.shape == y.shape == z.shape

        distance = (((np.array([x, y, z]).reshape(3, -1) - self.spacecraft.location.reshape(3, 1)) ** 2).sum(0) ** 0.5).reshape(-1)
        vx, vy = self.project_to_viewing_plane(x, y, z)
        px, py = self.coords_to_pixels(vx, vy)

        if not isinstance(color, np.ndarray):
            color = np.ones((distance.shape[0], 4)).astype(int) * prepare_color(color)

        if isinstance(radius, numbers.Number):
            radius = np.ones(distance.shape[0]).astype(int) * radius

        for i in np.argsort(distance)[::-1]:

            # This returns false if any of the terms are np.nan
            if all([px[i] > -radius[i], px[i] < self.Px + radius[i], py[i] > -radius[i], py[i] < self.Py + radius[i]]) :
                try:
                    t0 = time.time()
                    pygame.gfxdraw.filled_circle(self.canvas, int(px[i]), int(py[i]), int(radius[i] / distance[i]), color[i])
                    Tt += time.time() - t0
                except OverflowError:
                    print('overflow')


    def plot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, linewidth=2, color='white'):
        global Tt

        vx, vy = self.project_to_viewing_plane(x, y, z)
        px, py = self.coords_to_pixels(vx, vy)

        if not isinstance(color, np.ndarray):
            color = [prepare_color(color)] * x.shape[0]

        for i in range(len(x) - 1):

            try:
                px1, py1 = int(px[i]), int(py[i])
                px2, py2 = int(px[i + 1]), int(py[i + 1])
            except ValueError:
                continue

            try:
                t0 = time.time()
                pygame.draw.line(self.canvas, color[i], (px1, py1), (px2, py2), linewidth)
                Tt += time.time() - t0
            except OverflowError:
                pass

    def add_polygon(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, color: str='white'):
        global Tt

        vx, vy = self.project_to_viewing_plane(x, y, z)
        px, py = self.coords_to_pixels(vx, vy)

        points = np.array([px, py]).T

        if np.any(np.isnan(points)):
            pass
        else:
            t0 = time.time()
            pygame.gfxdraw.filled_polygon(self.canvas,points, color)
            Tt += time.time() - t0



    def add_object(self, obj: Object):
        self.objects.append(obj)


    def add_objects(self, objs: list):
        for obj in objs:
            self.add_object(obj)


    def update(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                sys.exit()

        self.spacecraft.update()

        pygame.display.update()
        self.canvas.fill(self.background_color)


        for obj in sorted(self.objects, key=lambda o: -((self.spacecraft.location - o.location) ** 2).sum()):


            if obj.plot_type == 'scatter':


                x, y, z, r, c = obj.get_points()
                x += obj.location[0]
                y += obj.location[1]
                z += obj.location[2]

                if obj.position == 'relative':
                    x += self.spacecraft.location[0]
                    y += self.spacecraft.location[1]
                    z += self.spacecraft.location[2]

                self.scatter(x, y, z, r, c)

            elif obj.plot_type == 'plot':
                x, y, z, r, c = obj.get_points()
                x += obj.location[0]
                y += obj.location[1]
                z += obj.location[2]

                self.plot(x, y, z, r, c)

            elif obj.plot_type == 'polygon':
                x, y, z, r, c = obj.get_points()
                x += obj.location[0]
                y += obj.location[1]
                z += obj.location[2]

                self.add_polygon(x, y, z, c)

            elif obj.plot_type == 'bitmap':

                squares, bitmap = obj.get_points()

                for s, b in zip(squares, bitmap):


                    x, y, z, c = s[0, :].copy(), s[1, :].copy(), s[2, :].copy(), prepare_color('white', alpha=b)

                    self.add_polygon(x, y, z, c)


            else:
                raise ValueError(f'Unknown plot type {obj.plot_type}')

        pygame.draw.line(self.canvas, (255, 255, 255, 255), (self.Px / 2 - 7, self.Py / 2), (self.Px / 2 + 7, self.Py / 2), 1)
        pygame.draw.line(self.canvas, (255, 255, 255, 255), (self.Px / 2, self.Py / 2 - 7), (self.Px / 2, self.Py / 2 + 7), 1)

        if self.FRAME_RATE is not None:
            self.clock.tick(self.FRAME_RATE)


class SpaceCraft:

    def __init__(self,
                 location: np.ndarray=None,
                 orientation: np.ndarray=None,
                 velocity: np.ndarray=None,
                 angular_velocity=None):


        if location is None:
            self.location = np.array([0, 0, 5], dtype=float)
        else:
            self.location = location.astype(float)

        if orientation is None:
            self.initital_orientation = np.array([0.05, 15, -0.98], dtype=float)
        else:
            self.initital_orientation = orientation.astype(float) / (orientation ** 2).sum() ** 0.5

        self.rotation_matrix = rotate(*self.initital_orientation)

        if velocity is None:
            self.velocity = np.zeros(3, dtype=float)
        else:
            self.velocity = velocity.astype(float)

        if angular_velocity is None:
            self.angular_velocity = np.zeros(3, dtype=float)
        else:
            self.angular_velocity = location.astype(float)

        self.speed = 0.001
        self.max_speed = 1
        self.force = np.zeros(3, dtype=float)
        self.drag = self.speed / (self.speed + self.max_speed) ** 3

        self.roll_speed = 0.0005
        self.max_roll_speed = 0.02
        self.angular_drag = self.roll_speed / (self.roll_speed + self.max_roll_speed) ** 3

    def direction(self) -> np.ndarray:
        """
        Find the unit vector which points in the direction the viewer is facing

        Returns
        -------
        n       The length-3 unit vector
        """

        return self.initital_orientation @ self.rotation_matrix

    def update(self):

        pressed_keys = pygame.key.get_pressed()

        # quit if we hit esc
        if pressed_keys[pygame.K_ESCAPE]:
            sys.exit()

        # forward:  nosedive forward
        if pressed_keys[pygame.K_UP]:
            self.angular_velocity[0] += self.roll_speed

        # backwards: pull upwards
        if pressed_keys[pygame.K_DOWN]:
            self.angular_velocity[0] -= self.roll_speed

        # right: turn right
        if pressed_keys[pygame.K_RIGHT]:
            self.angular_velocity[1] += self.roll_speed

        # left: turn left
        if pressed_keys[pygame.K_LEFT]:
            self.angular_velocity[1] -= self.roll_speed

        # d: roll right
        if pressed_keys[pygame.K_d]:
            self.angular_velocity[2] += self.roll_speed

        # a: roll left
        if pressed_keys[pygame.K_a]:
            self.angular_velocity[2] -= self.roll_speed


        # angular drag
        self.angular_velocity *=  1 - self.angular_drag * self.angular_velocity ** 2

        # angular correcting force
        correcting_torque = np.sign(self.angular_velocity) * self.roll_speed / 5
        correcting_torque[correcting_torque > np.abs(self.angular_velocity)] = 0
        self.angular_velocity -= correcting_torque

        self.rotation_matrix = rotate(*self.angular_velocity) @ self.rotation_matrix

        # w: accellerate in direction which you're pointing
        if pressed_keys[pygame.K_w]:
            self.velocity += self.speed * self.direction()

        # s: brake
        if pressed_keys[pygame.K_s]:
            self.velocity *= 0.95

        # velocity drag
        # self.velocity *=  1 - self.drag * self.velocity ** 2

        # correcting force
        correcting_force = np.sign(self.velocity) * self.speed / 5
        correcting_force[correcting_force > np.abs(self.velocity)] = 0
        self.velocity -= correcting_force

        self.location += self.velocity


if __name__ == '__main__':

    np.set_printoptions(precision=5, linewidth=500, threshold=1000, suppress=True, edgeitems=5)

    T = time.time()
    Tt = 0

    screen = Space(screen_pixels=(1920, 1080),
                   screen_coords=(5, 2.8125),
                   camera_initial_position=(0, 0, -5),
                   viewing_plane_distance=2,
                   add_stars=True,
                   frame_rate=30)

    ic = [(1, -1, 0), (1, 0, -1), (0, -1, 1)]

    for i in range(3):
        screen.add_object(LorenzAttractor(tail_length=1000, ititial_coords=ic[i]))

    while True:
        screen.update()


