import time
import numpy as np
import sys
import argparse
import os
import matplotlib.pyplot as plt

from planning import *


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
            edge = tree.get_edge(node1, node2)
            total_cost += edge.cost()

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
                cspace,
                qI,
                qG,
                self.edge_creator,
                self.distance_computator,
                self.collision_checker,

            )
            end_time = time.time()
            rrt_times.append(end_time - start_time)
            rrt_cost = G1.get_path_cost(
                self.qI, self.qG, self.distance_computator)
            print("rrt cost", rrt_cost)

            path1 = []
            if root1 is not None and goal1 is not None:
                path = G1.get_path(root1, goal1)
            rrt_cost = self.get_path_cost(G1, path1)

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

            path2 = []
            if root2 is not None and goal2 is not None:
                path = G2.get_path(root2, goal2)
            rrtstar_cost = self.get_path_cost(G2, path2)
            rrt_costs.append(rrtstar_cost)

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
