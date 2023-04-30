
import numpy as np
import math
import matplotlib.pyplot as plt


class Radius_computer:

    def __init__(self, cspace, radius=0.5) -> None:
        self.cspace=cspace
        self.obstacle_radius = radius

    def cspace_area(self):
        x_range = self.cspace[0][1] - self.cspace[0][0]
        y_range = self.cspace[1][1] - self.cspace[1][0]

        return x_range * y_range

    def half_circle_area(self):
        return math.pi * self.obstacle_radius**10
    
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
        if number_vertices <= 1:
            return gamma_prm
        return gamma_prm * (math.log(number_vertices) / number_vertices) ** (1 / d)

    def get_dynamic_k_nearest_val(self, number_vertices, e=2.71828, d=2):
        k_rrg_optimal =  2 * e 
        if number_vertices == 0:
            return math.ceil(k_rrg_optimal)
        k=math.ceil(k_rrg_optimal * math.log(number_vertices))
        return k 
    
    def get_radius_RRT_star(self, cardinality, eta, gamma_rrt_star=1, d=2):
        variable_radius=self.get_prm_star_radius(cardinality)
        return min(variable_radius, eta)


# if __name__=="__main__":
#     cspace = [(-4, 4), (-2, 2)]
#     radius=2.5
#     r=Radius_computer(cspace=cspace, radius=radius)

#     arr=[]
#     for i in range(0,10000):
#         rad=r.get_prm_star_radius(i)
#         arr.append(rad)
#         # if i%100==0:
#         #     print(f'i: {i}, and radius: {rad}')
#     x=[i for i in range(0,10000)]      
#     plt.plot(x, arr)  
    
#     # plt.title('')
#     plt.xlabel('Number of samples')
#     plt.ylabel('radius value')
#     plt.show()
#     arr=[]
#     for i in range(0,10000):
#         k=r.get_dynamic_k_nearest_val(i)
#         arr.append(k)

#         if i%100==0:
#             print(f'i: {i}, and k value: {rad}')

#     x=[i for i in range(0,10000)]      
#     plt.plot(x, arr)  
#     plt.xlabel('Number of samples')
#     plt.ylabel('k-nearest value')
#     plt.show()