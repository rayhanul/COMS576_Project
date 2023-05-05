

import pandas as pd
from bokeh.palettes import Category20  # Import the palette
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.palettes import Category20
import numpy as np
import math
import matplotlib.pyplot as plt


class Radius_computer:

    def __init__(self, cspace, radius=0.5) -> None:
        self.cspace = cspace
        self.obstacle_radius = radius

    def cspace_area(self):
        x_range = self.cspace[0][1] - self.cspace[0][0]
        y_range = self.cspace[1][1] - self.cspace[1][0]

        return x_range * y_range

    def half_circle_area(self, obs_count):
        return math.pi * self.obstacle_radius**obs_count

    def get_mu_x_free(self, obs_count):
        # print("self.cspace_area()", self.cspace_area(),
        #       "self.half_circle_area(obs_count)", self.half_circle_area(obs_count))
        return self.cspace_area() - self.half_circle_area(obs_count)

    def get_unit_ball_2_dimension(self):
        '''
        return the unit ball in 2-dimensional space-which is equal to pi 
        '''
        return math.pi

    def get_xi_d(self):
        return self.get_unit_ball_2_dimension()

    def get_gamma_prm(self, mu_xfree, xi_d, d=2):
        if mu_xfree < 0:
            mu_xfree = 0

        return (2*(1+(1/d))**(1/d)) * ((mu_xfree/xi_d)**(1/d))

    def get_prm_star_radius(self, number_vertices, obs_count=10, d=2, gamma_prm=1, r=float('inf'), ):
        """
        Calculate the radius for PRM*.

        :param n: int, number of samples (vertices in the graph)
        :param d: int, dimension of the space
        :param gamma: float, scaling factor (optional)
        :param r: float, constant value (optional)
        :return: float, PRM* radius
        """

        mu_x_free = self.get_mu_x_free(obs_count)
        xi_d = self.get_xi_d()

        gamma_prm_star = self.get_gamma_prm(mu_x_free, xi_d, d)
        gamma_prm = gamma_prm_star+0.1
        if number_vertices <= 1:
            return gamma_prm
        return gamma_prm * (math.log(number_vertices) / number_vertices) ** (1 / d)

    def get_dynamic_k_nearest_val(self, number_vertices, e=2.71828, d=2):
        k_rrg_optimal = 2 * e
        if number_vertices == 0:
            return math.ceil(k_rrg_optimal)
        k = math.ceil(k_rrg_optimal * math.log(number_vertices))
        return k

    def get_radius_RRT_star(self, cardinality, obs_count, eta, gamma_rrt_star=1, d=2):
        variable_radius = self.get_prm_star_radius(cardinality, obs_count)
        print(variable_radius, eta, cardinality)
        if cardinality == 0:
            return eta
        return min(variable_radius, eta)


# if __name__ == "__main__":
#     cspace = [(-4, 4), (-2, 2)]
#     radius = 2.5
#     r = Radius_computer(cspace=cspace, radius=radius)

#     x = [i for i in range(0, 10000)]

#     colors = Category20[11]

#     prm_star_fig = figure(title="PRM* radius", x_axis_label='Number of samples', y_axis_label='Radius value')
#     rrt_star_fig = figure(title="RRT* radius", x_axis_label='Number of samples', y_axis_label='Radius value')
#     k_nearest_fig = figure(title="k-nearest", x_axis_label='Number of samples', y_axis_label='k-nearest value')

#     for obs_count in range(11):
#         arr_prm_star = []
#         for i in range(0, 10000):
#             prm_star_rad = r.get_prm_star_radius(i, obs_count=obs_count)
#             arr_prm_star.append(prm_star_rad)

#         prm_star_fig.line(x, arr_prm_star, color=colors[obs_count], legend_label=f'{obs_count} Obstacles')

#         # Print PRM* radius values for the current obstacle count
#         print(f"PRM* radius values for {obs_count} obstacles: {arr_prm_star}")

#     for obs_count in range(11):
#         arr_rrt_star = []
#         for i in range(0, 10000):
#             rrt_star_rad = r.get_radius_RRT_star(i, obs_count, eta=2.5)
#             arr_rrt_star.append(rrt_star_rad)

#         rrt_star_fig.line(x, arr_rrt_star, color=colors[obs_count], legend_label=f'{obs_count} Obstacles')

#         # Print RRT* radius values for the current obstacle count
#         print(f"RRT* radius values for {obs_count} obstacles: {arr_rrt_star}")

#     for obs_count in range(11):
#         arr_k_nearest = []
#         for i in range(0, 10000):
#             k = r.get_dynamic_k_nearest_val(i)
#             arr_k_nearest.append(k)

#         k_nearest_fig.line(x, arr_k_nearest, color=colors[obs_count], legend_label=f'{obs_count} Obstacles')

#         # Print k-nearest values for the current obstacle count
#         print(f"k-nearest values for {obs_count} obstacles: {arr_k_nearest}")

#     grid = gridplot([[prm_star_fig, rrt_star_fig, k_nearest_fig]])

#     show(grid)


# ...
if __name__ == "__main__":
    # ...
    cspace = [(-4, 4), (-2, 2)]
    radius = 2.5
    r = Radius_computer(cspace=cspace, radius=radius)

    x = [i for i in range(0, 10000)]

    colors = Category20[11]

    prm_star_fig = figure(
        title="PRM* radius", x_axis_label='Number of samples', y_axis_label='Radius value')
    rrt_star_fig = figure(
        title="RRT* radius", x_axis_label='Number of samples', y_axis_label='Radius value')
    k_nearest_fig = figure(
        title="k-nearest", x_axis_label='Number of samples', y_axis_label='k-nearest value')

    # Create a dataframe to store the results
    columns = ["Algorithm", "Obstacle Count", "Value"]
    results = []

    # ...

    for obs_count in range(11):
        arr_prm_star = []
        for i in range(0, 10000):
            prm_star_rad = r.get_prm_star_radius(i, obs_count=obs_count)
            arr_prm_star.append(prm_star_rad)
        average_prm_star = sum(arr_prm_star) / len(arr_prm_star)
        results.append({"Algorithm": "PRM* Radius",
                        "Obstacle Count": obs_count,
                        "Value": average_prm_star})

    # ...

    for obs_count in range(11):
        arr_rrt_star = []
        for i in range(0, 10000):
            rrt_star_rad = r.get_radius_RRT_star(i, obs_count, eta=2.5)
            arr_rrt_star.append(rrt_star_rad)
        average_rrt_star = sum(arr_rrt_star) / len(arr_rrt_star)
        results.append({"Algorithm": "RRT* Radius",
                        "Obstacle Count": obs_count,
                        "Value": average_rrt_star})

    # ...

    for obs_count in range(11):
        arr_k_nearest = []
        for i in range(0, 10000):
            k = r.get_dynamic_k_nearest_val(i)
            arr_k_nearest.append(k)
        average_k_nearest = sum(arr_k_nearest) / len(arr_k_nearest)
        results.append({"Algorithm": "k-nearest",
                        "Obstacle Count": obs_count,
                        "Value": average_k_nearest})

    # ...

    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(results)

    # Print the results dataframe
    print(results_df)
