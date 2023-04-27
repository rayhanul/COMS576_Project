from obstacle import construct_circular_obstacles, WorldBoundary2D
import numpy as np
import math


class radius_computer:

    import math

    def cspace_area(cspace):
        x_range = cspace[0][1] - cspace[0][0]
        y_range = cspace[1][1] - cspace[1][0]
        return x_range * y_range

    def half_circle_area(radius):
        return 0.5 * math.pi * radius**2

    import math

    def prm_star_radius(self, n, d, gamma=1, r=float('inf')):
        """
        Calculate the radius for PRM*.

        :param n: int, number of samples (vertices in the graph)
        :param d: int, dimension of the space
        :param gamma: float, scaling factor (optional)
        :param r: float, constant value (optional)
        :return: float, PRM* radius
        """
        return min(r, gamma * (math.log(n) / n) ** (1 / d))


if __name__ == '__main__':
    print("This is test")

    cspace = [(-3, 3), (-1, 1)]

    x = radius_computer()

    cspace = [(-3, 3), (-1, 1)]
    obstacle_radius = 0.98
    obstacle_center = (0, 0)

    total_area = x.cspace_area(cspace)
    obstacle_area = x.half_circle_area(obstacle_radius)
    obstacle_free_area = total_area - obstacle_area

    print("Area of the obstacle-free space:", obstacle_free_area)
    # Calculate radius for PRM*
    num_samples = 1000
    dimension = 2
    radius = x.prm_star_radius(num_samples, dimension)

print("Radius for PRM*:", radius)
