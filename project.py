import sys
import argparse
import os
import matplotlib.pyplot as plt
from planning import (
    rrt,
    rrt_star,
    prm,
    prm_star,
    StraightEdgeCreator,
    EuclideanDistanceComputator,
    EmptyCollisionChecker,
    ObstacleCollisionChecker,
)
from obstacle import construct_circular_obstacles, WorldBoundary2D
from draw_cspace import draw

from radius_computer import Radius_computer

ALG_RRT = "rrt"
ALG_PRM = "prm"
ALG_PRM_STAR = "prm_star"
ALG_RRT_STAR = "rrt_star"


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run sampling-based motion planning algorithm"
    )
    parser.add_argument(
        "--alg",
        choices=[ALG_RRT, ALG_PRM, ALG_PRM_STAR, ALG_RRT_STAR],
        required=False,
        default=ALG_RRT,
        dest="alg",
        help="algorithm, default to rrt",
    )
    parser.add_argument(
        "--type",
        choices=['k', 'r'],
        required=False,
        default='k',
        dest="type",
        help="algorithm, default to use k nearest",
    )
    args = parser.parse_args(sys.argv[1:])
    return args


def main_rrt(
    cspace, qI, qG, edge_creator, distance_computator, collision_checker
):
    """Task 1 (Exploring the C-space using RRT) and Task 2 (Solve the planning problem using RRT)"""
    fig, ax3 = plt.subplots()
    title3 = "RRT planning"
    (G3, root3, goal3) = rrt(
        cspace=cspace,
        qI=qI,
        qG=qG,
        edge_creator=edge_creator,
        distance_computator=distance_computator,
        collision_checker=collision_checker,

    )

    path = []
    if goal3 is not None:
        path = G3.get_path(root3, goal3)
    draw(ax3, cspace, obs_boundaries, qI, qG, G3, path, title3)

    plt.show()


def main_rrt_star(
    cspace, qI, qG, edge_creator, distance_computator, collision_checker, radius_computer, k_nearest
):
    """Task 1 (Exploring the C-space using RRT) and Task 2 (Solve the planning problem using RRT)"""
    fig, ax3 = plt.subplots()
    approach = "K Nearest" if k_nearest else "Radius"
    title3 = f"RRT Star planning with {approach}"
    (G3, root3, goal3) = rrt_star(
        cspace=cspace,
        qI=qI,
        qG=qG,
        edge_creator=edge_creator,
        distance_computator=distance_computator,
        collision_checker=collision_checker,
        radius_computer=radius_computer,
        k_nearest=k_nearest,
    )
    path = []
    if goal3 is not None:
        path = G3.get_path(root3, goal3)
    draw(ax3, cspace, obs_boundaries, qI, qG, G3, path, title3)

    plt.show()


def main_prm(
    cspace, qI, qG, edge_creator, distance_computator, collision_checker, obs_boundaries,
):
    """Task 3 (Solve the planning problem using PRM)"""
    fig, ax = plt.subplots()
    title = "PRM planning"
    (G, root, goal) = prm(
        cspace=cspace,
        qI=qI,
        qG=qG,
        edge_creator=edge_creator,
        distance_computator=distance_computator,
        collision_checker=collision_checker,
    )
    path = []
    if root is not None and goal is not None:
        path = G.get_path(root, goal)
    draw(ax, cspace, obs_boundaries, qI, qG, G, path, title)
    plt.show()


def main_prm_star(
    cspace, qI, qG, edge_creator, distance_computator, collision_checker, radius_computer, obs_boundaries, k_nearest_prm_star
):
    """Task 3 (Solve the planning problem using PRM)"""
    fig, ax = plt.subplots()
    approach = "K Nearest" if k_nearest else "Radius"
    title = f"PRM Star planning with {approach}"
    (G, root, goal) = prm_star(
        cspace=cspace,
        qI=qI,
        qG=qG,
        edge_creator=edge_creator,
        distance_computator=distance_computator,
        collision_checker=collision_checker,
        radius_computer=radius_computer,
        obs_boundaries=obs_boundaries,
        k_nearest_prm_star=k_nearest_prm_star
    )
    path = []
    if root is not None and goal is not None:
        path = G.get_path(root, goal)
    draw(ax, cspace, obs_boundaries, qI, qG, G, path, title)
    plt.show()


if __name__ == "__main__":
    # python hw4.py --alg rrt
    # sys.argv = [os.path.basename(__file__), '--alg', 'rrt_star', '--type', 'k']

    cspace = [(-4, 4), (-2, 2)]
    qI = (-3, -0.5)
    qG = (3, 1)
    obstacles = construct_circular_obstacles(0.6)
    obs_boundaries = [obstacle.get_boundaries() for obstacle in obstacles]

    # We don't really need to explicitly need to check the world boundary
    # because the world is convex and we're connecting points by a straight line.
    # So as long as the two points are within the world, the line between them
    # are also within the world.
    # I'm just including this for completeness.
    world_boundary = WorldBoundary2D(cspace[0], cspace[1])
    obstacles.append(world_boundary)

    edge_creator = StraightEdgeCreator(0.1)
    collision_checker = ObstacleCollisionChecker(obstacles)
    distance_computator = EuclideanDistanceComputator()
    radius_computer = Radius_computer(cspace=cspace, raidus=0.98)

    args = parse_args()
    k_nearest = True if args.type == 'k' else False

    if args.alg == ALG_RRT:
        main_rrt(cspace, qI, qG, edge_creator,
                 distance_computator, collision_checker,)
    elif args.alg == ALG_PRM_STAR:
        main_prm_star(cspace, qI, qG, edge_creator, distance_computator,
                      collision_checker, radius_computer, obs_boundaries, k_nearest,)
    elif args.alg == ALG_RRT_STAR:
        main_rrt_star(cspace, qI, qG, edge_creator, distance_computator,
                      collision_checker, radius_computer, k_nearest,)
    else:
        main_prm(cspace, qI, qG, edge_creator, distance_computator,
                 collision_checker, obs_boundaries,)
