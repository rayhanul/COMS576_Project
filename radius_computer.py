
import numpy as np
import math


class Radius_computer:

    def __init__(self, cspace, raidus) -> None:
        self.cspace=cspace
        self.obstacle_radius = raidus

    def cspace_area(self):
        x_range = self.cspace[0][1] - self.cspace[0][0]
        y_range = self.cspace[1][1] - self.cspace[1][0]

        return x_range * y_range

    def half_circle_area(self):
        return 0.5 * math.pi * self.obstacle_radius**2
    
    def get_mu_x_free(self):
        return self.cspace_area() - self.half_circle_area()
    
    def get_unit_ball_2_dimension(self):
        '''
        return the unit ball in 2-dimensional space-which is equal to pi 
        '''
        return math.pi 

    def get_xi_d(self):
        return self.get_unit_ball_2_dimension()

    def get_gamma_prm(self, mu_xfree, xi_d, d=2):

        return (2*(1+(1/d))**(1/d)) * ((mu_xfree/xi_d)**(1/d))
    
    def get_prm_star_radius(self, number_vertices, d=2, gamma_prm=1, r=float('inf')):
        """
        Calculate the radius for PRM*.

        :param n: int, number of samples (vertices in the graph)
        :param d: int, dimension of the space
        :param gamma: float, scaling factor (optional)
        :param r: float, constant value (optional)
        :return: float, PRM* radius
        """


        mu_x_free=self.get_mu_x_free()
        xi_d=self.get_xi_d()

        gamma_prm_star=self.get_gamma_prm(mu_x_free, xi_d, d)
        gamma_prm=gamma_prm_star+0.1
        if number_vertices==0:
            return 0
        return min(r, gamma_prm * (math.log(number_vertices) / number_vertices) ** (1 / d))

    def get_k_prm_star(self, number_vertices, e=2.71828, d=2):
        k_prm =  2 * e 
        if number_vertices == 0:
            return 0
        return math.ceil(k_prm * math.log(number_vertices))
    
    def get_k_nearest_RRT_star(self, cardinality, k_rrg=5.6):
        if cardinality ==0:
            return 0
        return math.ceil(k_rrg * math.log(cardinality))
    
    def get_radius_RRT_star(self, cardinality, gamma_rrt_star=1, eta=2, d=2):

        variable_radius=gamma_rrt_star * (math.log(cardinality)/ cardinality)**(1/d)
        return min(variable_radius, eta)

# if __name__ == '__main__':
#     print("This is test")

#     cspace = [(-3, 3), (-1, 1)]

#     x = Radius_computer(cspace, 0.98)

#     cspace = [(-3, 3), (-1, 1)]
    
#     obstacle_center = (0, 0)

#     total_area = x.cspace_area()
#     obstacle_area = x.half_circle_area()
#     obstacle_free_area = total_area - obstacle_area


#     # Calculate radius for PRM*
#     num_samples = 1000
#     dimension = 2
#     for i in range(0, 1000):
#         radius = x.get_prm_star_radius(i)
#         if i%100==0:
#             print("Radius for PRM*:", radius)
