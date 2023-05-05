from radius_computer import Radius_computer
import seaborn as sns
import time
import numpy as np
import sys
import argparse
import os
import matplotlib.pyplot as plt

from planning import *

import numpy as np
import pandas as pd
# Set the number of iterations
num_iterations = 20


class Plotter:

    def __init__(self, cspace, qI, qG, edge_creator, distance_computator, collision_checker, radius_computer, obs_boundaries, k_nearest):
        self.cspace = cspace
        self.qI = qI
        self.qG = qG
        self.edge_creator = edge_creator
        self.distance_computator = distance_computator
        self.collision_checker = collision_checker
        self.radius_computer = radius_computer
        self.obs_boundaries = obs_boundaries
        self.k_nearest_prm_star = k_nearest
        self.k_nearest = k_nearest

    # Create a function to calculate path length

    def comparision_time(self):

        # Set up the test environment
        rrt_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                     self.distance_computator, self.collision_checker,)
        rrt_star_r_param = (self.cspace, self.qI, self.qG, self.edge_creator, self.distance_computator,
                            self.collision_checker, self.radius_computer, self.k_nearest,)
        rrt_star_k_param = (self.cspace, self.qI, self.qG, self.edge_creator, self.distance_computator,
                            self.collision_checker, self.radius_computer, self.k_nearest,)
        prm_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                     self.distance_computator, self.collision_checker,)
        prm_star_r_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                            self.distance_computator, self.collision_checker, self.radius_computer, self.obs_boundaries, self.k_nearest_prm_star,)
        prm_star_k_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                            self.distance_computator, self.collision_checker, self.radius_computer, self.obs_boundaries, self.k_nearest_prm_star,)

        # Initialize result data structures
        results = []

        # Run the algorithms
        for i in range(num_iterations):
            for algo_name, algo_func, algo_params in zip(['RRT', 'RRT*(r)', 'RRT*(k)', 'PRM', 'PRM*(r)', 'PRM*(k)'], [rrt, rrt_star, rrt_star, prm, prm_star, prm_star], [rrt_param, rrt_star_r_param, rrt_star_k_param, prm_param, prm_star_r_param, prm_star_k_param]):
                start_time = time.time()

                (G, root, goal) = algo_func(*algo_params)

                vertex_path = G.get_vertex_path(root, goal)
                path_cost = G.get_path_cost(vertex_path)
                end_time = time.time()
                results.append(
                    {'Iteration': i, 'Algorithm': algo_name, 'Path Time': (end_time-start_time), 'Path Cost': path_cost})
                print(end_time-start_time, path_cost)

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        print(results_df.groupby(
            ['Algorithm'])['Path Time'].mean())

        print(results_df.groupby(
            ['Algorithm'])['Path Cost'].mean())

    def comparision_vertices(self):
        # Set the range of iterationss for the comparison
        iterations = [100, 200, 300, 400, 500]

        # Initialize result data structures
        results = []
        print('here')
        # Run the algorithms
        for k in iterations:
            # for radius in radius_values:
            # radius_computer = Radius_computer(
            #     cspace=self.cspace, radius=radius)
            # Update the RRT* and PRM* parameter dictionaries with the current k and radius values
            rrt_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                         self.distance_computator, self.collision_checker, 0.1, k,)
            rrt_star_param_k = (self.cspace, self.qI, self.qG, self.edge_creator, self.distance_computator,
                                self.collision_checker, self.radius_computer, True, 20, k)
            rrt_star_param_r = (self.cspace, self.qI, self.qG, self.edge_creator, self.distance_computator,
                                self.collision_checker, self.radius_computer, False, 20, k)

            for i in range(num_iterations):
                for algo_name, algo_func, algo_params in zip(['RRT', 'RRT*(r)', 'RRT*(k)'], [rrt, rrt_star, rrt_star], [rrt_param, rrt_star_param_k, rrt_star_param_r]):
                    (G, root, goal) = algo_func(*algo_params)
                    path_length = G.get_path_cost(
                        self.qI, self.qG, self.distance_computator)
                    print(algo_name, path_length)
                    results.append(
                        {'Iteration': i, 'Algorithm': algo_name, 'vertices': len(G.vertices), 'iterations': k, })
                    print(k, i)

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        print(results_df.groupby(
            ['Algorithm', 'iterations'])['vertices'].mean())
        # print(results_df)

    def calculate_path_length(self, path):
        path_length = 0
        for i in range(len(path) - 1):
            path_length += euclidean_distance(path[i], path[i+1])
        return path_length

    def comparision_path_length(self):

        # Set up the test environment
        rrt_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                     self.distance_computator, self.collision_checker,)
        rrt_star_r__param = (self.cspace, self.qI, self.qG, self.edge_creator, self.distance_computator,
                             self.collision_checker, self.radius_computer, self.k_nearest,)
        rrt_star__k_param = (self.cspace, self.qI, self.qG, self.edge_creator, self.distance_computator,
                             self.collision_checker, self.radius_computer, self.k_nearest,)
        prm_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                     self.distance_computator, self.collision_checker,)
        prm_star_r_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                            self.distance_computator, self.collision_checker, self.radius_computer, self.obs_boundaries, self.k_nearest_prm_star,)
        prm_star_k_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                            self.distance_computator, self.collision_checker, self.radius_computer, self.obs_boundaries, self.k_nearest_prm_star,)

        # Initialize result data structures
        results = []

        # Run the algorithms
        for i in range(num_iterations):
            for algo_name, algo_func, algo_params in zip(['RRT', 'RRT*(r)', 'RRT*(k),' 'PRM', 'PRM*(r)', 'PRM*(k)'], [rrt, rrt_star, rrt_star, prm, prm_star, prm_star], [rrt_param, rrt_star_r__param, prm_param, prm_star_r_param, prm_star_k_param]):
                (G, root, goal) = algo_func(*algo_params)
                path_length = G.get_path_cost(
                    self.qI, self.qG, self.distance_computator)
                results.append(
                    {'Iteration': i, 'Algorithm': algo_name, 'Path Length': path_length})

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Calculate the moving average path length for each algorithm
        results_df['Moving Average Path Length'] = results_df.groupby('Algorithm')[
            'Path Length'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())

        # Plot individual path lengths for each algorithm
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Iteration', y='Path Length', hue='Algorithm',
                     style='Algorithm', markers=True, dashes=False, data=results_df, ax=ax1)
        ax1.set_title('Individual Path Lengths')
        plt.legend(title='Algorithm')

        # Plot moving average path lengths for each algorithm
        fig, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Iteration', y='Moving Average Path Length', hue='Algorithm',
                     style='Algorithm', markers=True, dashes=False, data=results_df, ax=ax2)
        ax2.set_title('Moving Average Path Lengths')
        plt.legend(title='Algorithm')

        plt.show()

    def comparision_k_R(self):
        # Set the range of k values and radius values for the comparison
        # Replace with a list of k values to test
        k_values = np.random.randint(1, 50, 10)
        # Replace with a list of radius values to test
        radius_values = np.random.uniform(0.1, 3, 20)

        # Initialize result data structures
        results = []
        print('here')
        # Run the algorithms
        for k in k_values:
            # for radius in radius_values:
            # radius_computer = Radius_computer(
            #     cspace=self.cspace, radius=radius)
            # Update the RRT* and PRM* parameter dictionaries with the current k and radius values
            rrt_star_param = (self.cspace, self.qI, self.qG, self.edge_creator, self.distance_computator,
                              self.collision_checker, self.radius_computer, True, k)
            prm_star_param = (self.cspace, self.qI, self.qG, self.edge_creator,
                              self.distance_computator, self.collision_checker, self.radius_computer, self.obs_boundaries, True, k,)

            for i in range(num_iterations):
                for algo_name, algo_func, algo_params in zip(['RRT*', 'PRM*'], [rrt_star, prm_star], [rrt_star_param, prm_star_param]):
                    (G, root, goal) = algo_func(*algo_params)
                    path_length = G.get_path_cost(
                        self.qI, self.qG, self.distance_computator)
                    print(algo_name, path_length)
                    results.append(
                        {'Iteration': i, 'Algorithm': algo_name, 'Path Length': path_length, 'k': k, })
                    print(k, i)
        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        # Calculate the moving average path length for each algorithm
        results_df['Moving Average Path Length'] = results_df.groupby(['Algorithm', 'k'])[
            'Path Length'].transform(lambda x: x.rolling(window=1, min_periods=1).mean())

        # Plot individual path lengths for each algorithm
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Iteration', y='Path Length', hue='Algorithm', style='Algorithm',
                     markers=True, dashes=False, data=results_df, ax=ax1)
        ax1.set_title('Individual Path Lengths')
        plt.legend(title='Algorithm')

        # Plot moving average path lengths for each algorithm
        fig, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Iteration', y='Moving Average Path Length', hue='Algorithm', style='Algorithm',
                     markers=True, dashes=False, data=results_df, ax=ax2)
        ax2.set_title('Moving Average Path Lengths')
        plt.legend(title='Algorithm')

        plt.show()

    def get_path_cost(self, tree, path):
        """Compute the cost of a given path in the tree.

        @type tree: a Tree object representing the search tree.
        @type path: a list of node IDs in the tree, representing a path.

        @return: a float, the total cost of the path.
        """
        total_cost = 0.0
        for i in range(len(path) - 1):
            node1 = path[i]
            node2 = path[i + 1]
            edge = tree.get_edge_cost(node1, node2)
            total_cost += edge
            print("edge cost ", edge)

        return total_cost

    def time_cost_analysis(self):  # Set up parameters for the algorithms
        print("inside")

        cspace = [(-4, 4), (-2, 2)]
        qI = (-3, -0.5)
        qG = (3, 1)

        numIt = 1000
        tol = 1e-3
        d = len(cspace)
        gamma_rrt = 1.0
        eta_rrt = 0.1
        gamma_rrt_star = 1.0
        eta_rrt_star = 2.5
        k_nearest_prm_star = False
        gamma_prm = 1.0
        eta_prm = 2.5

        # Initialize time and cost lists
        rrt_times = []
        rrt_costs = []
        rrt_star_times = []
        rrt_star_costs = []
        prm_times = []
        prm_costs = []
        prm_star_times = []
        prm_star_costs = []

        # Run each algorithm multiple times and record the times and costs

        for i in range(10):
            # RRT
            start_time = time.time()
            (G1, root1, goal1) = rrt(
                cspace=self.cspace,
                qI=self.qI,
                qG=self.qG,
                edge_creator=self.edge_creator,
                distance_computator=self.distance_computator,
                collision_checker=self.collision_checker,

            )

            vertex_set_to_goal = []
            if goal1 is not None:
                vertex_set_to_goal = G1.get_vertices_path_to_goal(root1, goal1)
            print("goal3", vertex_set_to_goal)
            print("path", vertex_set_to_goal)
            end_time = time.time()
            rrt_times.append(end_time - start_time)
            # rrt_cost = G1.get_path_cost(
            #     self.qI, self.qG, self.distance_computator)
            # print("rrt cost", rrt_cost)

            rrt_cost = self.get_path_cost(G1, vertex_set_to_goal)

            rrt_costs.append(rrt_cost)

            # RRT*
            start_time = time.time()
            (G2, root2, goal2) = rrt_star(
                cspace,
                qI,
                qG,
                self.edge_creator,
                self.distance_computator,
                self.collision_checker,
                self.radius_computer,
                self.k_nearest_prm_star,
            )
            end_time = time.time()
            rrt_star_times.append(end_time - start_time)

            vertex_set_to_goal = []
            if goal2 is not None:
                vertex_set_to_goal = G2.get_vertices_path_to_goal(root2, goal2)
            print("goal3", vertex_set_to_goal)
            print("path", vertex_set_to_goal)
            rrt_star_cost = self.get_path_cost(G2, vertex_set_to_goal)

            rrt_costs.append(rrt_star_cost)

            # PRM
            start_time = time.time()
            (G3, root3, goal3) = prm(
                cspace,
                qI,
                qG,
                self.edge_creator,
                self.distance_computator,
                self.collision_checker,
            )
            end_time = time.time()
            prm_times.append(end_time - start_time)
            prm_costs.append(G3.get_path_cost(
                self.qI, self.qG, self.distance_computator))

            # PRM*
            start_time = time.time()
            (G4, root4, goal4) = prm_star(
                self.cspace,
                self.qI,
                self.qG,
                self.edge_creator,
                self.distance_computator,
                self.collision_checker,
                self.radius_computer,
                self.obs_boundaries,
                self.k_nearest_prm_star
            )
            end_time = time.time()
            prm_star_times.append(end_time - start_time)
            prm_star_costs.append(G4.get_path_cost(
                self.qI, self.qG, self.distance_computator))

        # Print out the average times and costs for each algorithm
        print("RRT:")
        print("Average time:", np.mean(rrt_times))
        print("Average cost:", np.mean(rrt_costs))
        print("")

        print("RRT*:")
        print("Average time:", np.mean(rrt_star_times))
        print("Average cost:", np.mean(rrt_star_costs))
        print("")

        print("PRM:")
        print("Average time:", np.mean(prm_times))
        print("Average cost:", np.mean(prm_costs))
        print("")

        print("PRM*:")
        print("Average time:", np.mean(prm_star_times))
        print("Average cost:", np.mean(prm_star_costs))
