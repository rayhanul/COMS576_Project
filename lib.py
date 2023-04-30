def compute_obstacle_volume(self,flag='circle',rad=0.02):
        obstacles=construct_circular_obstacles(rad)
        obstacle_area = 0
        for obs in obstacles:
            if flag == "circle": # Check if obstacle is a circle
                radius = rad
                area = np.pi * radius**2
                obstacle_area += area / 2  # Divide area by 2 for a half-circle
            else:
                # Handle other obstacle shapes (e.g., rectangles, polygons)
                pass
    
        return obstacle_area
    
    def compute_free_space_volume(self,workspace_bounds):
        d = len(workspace_bounds) # dimension of the workspace
    
        # Compute the volume of the workspace
        workspace_volume = np.prod(workspace_bounds[:, 1] - workspace_bounds[:, 0])
    
        # Compute the total volume of the obstacles
        obstacle_volume = self.compute_obstacle_volume()
    
        # Compute the volume of the free space
        free_space_volume = workspace_volume - obstacle_volume
    
        return free_space_volume
    def leb(self):
        # Calculate the dimension of the configuration space
        Xfree=self.compute_free_space_volume()
        d = Xfree.shape[1]
    
        # Calculate the Lebesgue measure of the obstacle-free space
        mu_Xfree = np.prod(np.max(Xfree, axis=0) - np.min(Xfree, axis=0))
    
        # Calculate the volume of the unit ball in d-dimensional Euclidean space
        zeta_d = np.pi**(d/2) / np.math.gamma(d/2 + 1)
    
        # Calculate the constant gamma_PRM
        gamma_PRM = 2 * (1 + 1/d) * (mu_Xfree / zeta_d)**(1/d)
    
        # Calculate the connection radius as a function of n
        r_n = gamma_PRM * (np.log() / len(self.vertices))**(1/d)
    
        # Generate random samples in the free configuration space
        #X = np.random.uniform(low=np.min(Xfree, axis=0), high=np.max(Xfree, axis=0), size=(n_samples, d))
        return r_n



class Tree(Graph):
    """A graph where each vertex has at most one parent"""

    def __init__(self):
        super().__init__()
        self.vertex_costs = {}  # Add a dictionary to store the cost-to-come for each vertex

    def add_edge(self, vid1, vid2, edge):
        """Add an edge from vertex with id vid1 to vertex with id vid2"""
        # Ensure that a vertex only has at most one parent (this is a tree).
        assert len(self.parents[vid2]) == 0
        super().add_edge(vid1, vid2, edge)

    def get_vertex_path(self, root_vertex, goal_vertex):
        """Trace back parents to return a path from root_vertex to goal_vertex"""
        vertex_path = [goal_vertex]
        v = goal_vertex
        while v != root_vertex:
            parents = self.parents[v]
            if len(parents) == 0:
                return []
            v = parents[0]
            vertex_path.insert(0, v)
        return vertex_path

    def set_vertex_cost(self, vid, cost):
        """Set the cost-to-come for the vertex with id vid"""
        self.vertex_costs[vid] = cost

    def get_vertex_cost(self, vid):
        """Set the cost-to-come for the vertex with id vid"""
        if vid not in self.vertex_costs:
            return 0.0
        return self.vertex_costs[vid]

    def get_nearby_vertices(self, state, radius, distance_computator):
        """Return the ids of vertices within radius of the given state based on the given distance function"""
        nearby_vertices = [
            vid for vid, s in self.vertices.items() if distance_computator.get_distance(s, state) <= radius
        ]
        return nearby_vertices

    def get_vertex_parent(self, vid):
        """Get the parent of a vertex with id vid"""
        parents = self.parents.get(vid)
        if parents:
            return parents[0]
        return None

    def remove_edge(self, edge_id):
        """Remove a given edge

        @type edge: a tuple (vid1, vid2) indicating the id of the origin and the destination vertices
        """
        super().remove_edge(edge_id)



def rrt_star(
    cspace,
    qI,
    qG,
    edge_creator,
    distance_computator,
    collision_checker,
    pG=0.1,
    numIt=100,
    tol=1e-3,
    k=10,
):
    def rewire(G, vs, distance_computator, edge_creator, collision_checker, k):
        qs = G.get_vertex_state(vs)
        vertices = G.get_nearest_vertices(qs, k, distance_computator)
        for vn in vertices:
            if vn != vs:
                qn = G.get_vertex_state(vn)
                (qe, edge) = stopping_configuration(
                    qs, qn, edge_creator, collision_checker, tol
                )
                if qe is not None and get_euclidean_distance(qn, qe) < tol:
                    cost_to_come = G.get_vertex_cost(vs) + edge.get_cost()
                    if cost_to_come < G.get_vertex_cost(vn):
                        G.set_parent(vs, vn, edge)
                        G.set_vertex_cost(vn, cost_to_come)

                        # Update cost-to-come of all children of vn
                        queue = [vn]
                        while queue:
                            u = queue.pop(0)
                            for v in G.parents.keys():
                                if u in G.parents[v]:
                                    cost_to_come = G.get_vertex_cost(
                                        u) + G.get_edge_cost(u, v)
                                    G.set_vertex_cost(v, cost_to_come)
                                    queue.append(v)

    G = Tree()
    root = G.add_vertex(np.array(qI))
    G.set_vertex_cost(root, 0)
    for i in range(numIt):
        use_goal = qG is not None and random.uniform(0, 1) <= pG
        if use_goal:
            alpha = np.array(qG)
        else:
            alpha = sample(cspace)
        vn = G.get_nearest(alpha, distance_computator, tol)
        qn = G.get_vertex_state(vn)
        (qs, edge) = stopping_configuration(
            qn, alpha, edge_creator, collision_checker, tol
        )
        if qs is None or edge is None:
            continue
        dist = get_euclidean_distance(qn, qs)
        if dist > tol:
            vs = G.add_vertex(qs)
            print("vs", vs)
            G.add_edge(vn, vs, edge)
            G.set_vertex_cost(vs, G.get_vertex_cost(vn) + edge.get_cost())
            rewire(G, vs, distance_computator,
                   edge_creator, collision_checker, k)
            if use_goal and get_euclidean_distance(qs, qG) < tol:
                return (G, root, vs)

    return (G, root, None)




import math
import copy
from heapq import heappush
from myQueue import QueueAstar


class Graph:
    """A class for maintaining a graph"""

    def __init__(self):
        # a dictionary whose key = id of the vertex and value = state of the vertex
        self.vertices = {}

        # a dictionary whose key = id of the vertex and value is the list of the ids of
        # its parents
        self.parents = {}

        # a dictionary whose key = (v1, v2) and value = (cost, edge).
        # v1 is the id of the origin vertex and v2 is the id of the destination vertex.
        # cost is the cost of the edge.
        # edge is of type Edge and stores information about the edge, e.g.,
        # the origin and destination states and the discretized points along the edge
        self.edges = {}

    def __str__(self):
        return "vertices: " + str(self.vertices) + " edges: " + str(self.edges)

    def add_vertex(self, state):
        """Add a vertex at a given state

        @return the id of the added vertex
        """
        vid = len(self.vertices)
        self.vertices[vid] = state
        self.parents[vid] = []
        return vid

    def get_vertex_state(self, vid):
        """Get the state of the vertex with id = vid"""
        return self.vertices[vid]

    def get_vertices(self):
        return list(self.vertices.keys())

    def add_edge(self, vid1, vid2, edge):
        """Add an edge from vertex with id vid1 to vertex with id vid2"""
        self.edges[(vid1, vid2)] = (
            edge.get_cost(),
            edge,
        )
        self.parents[vid2].append(vid1)

    def remove_edge(self, edge_id):
        """Remove a given edge

        @type edge: a tuple (vid1, vid2) indicating the id of the origin and the destination vertices
        """
        del self.edges[edge_id]
        v1 = edge_id[0]
        v2 = edge_id[1]
        self.parents[v2].remove(v1)

    def get_nearest(self, state, distance_computator, tol):
        """Return the vertex in the swath of the graph that is closest to the given state"""

        if len(self.edges) == 0:
            return self.get_nearest_vertex(state, distance_computator)

        (nearest_edge, nearest_t) = self.get_nearest_edge(
            state, distance_computator)
        if nearest_t <= tol:
            return nearest_edge[0]

        if nearest_t >= 1 - tol:
            return nearest_edge[1]

        return self.split_edge(nearest_edge, nearest_t)

    def get_nearest_edge(self, state, distance_computator):
        """Return the edge that is nearest to the given state based on the given distance function
        @type distance_computator: a DistanceComputator object that includes the get_distance(s1, s2)
            function, which returns the distance between s1 and s2.

        @return a tuple (nearest_edge, nearest_t) where
            * nearest_edge is a tuple (vid1, vid2), indicating the id of the origin and the destination vertices
            * nearest_t is a float in [0, 1], such that the nearest point along the edge to the given state is at
              distance nearest_t/length where length is the length of nearest_edge
        """
        nearest_dist = math.inf
        nearest_edge = None
        nearest_t = None

        for edge_id, (cost, edge) in self.edges.items():
            (sstar, tstar) = edge.get_nearest_point(state)
            dist = distance_computator.get_distance(sstar, state)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_edge = edge_id
                nearest_t = tstar

        return (nearest_edge, nearest_t)

    def get_nearest_vertex(self, state, distance_computator):
        """Return the id of the nearest vertex to the given state based on the given distance function
        @type distance_computator: a DistanceComputator object that includes the get_distance(s1, s2)
            function, which returns the distance between s1 and s2.
        """
        nearest_dist = math.inf
        nearest_vertex = None
        for vertex, s in self.vertices.items():
            dist = distance_computator.get_distance(s, state)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_vertex = vertex
        return nearest_vertex

    def get_nearest_vertices(self, state, k, distance_computator, PRM_star=0):
        """Return the ids of k nearest vertices to the given state based on the given distance function
        @type distance_computator: a DistanceComputator object that includes the get_distance(s1, s2)
            function, which returns the distance between s1 and s2.
        """
        dist_vertices = []
        for vertex, s in self.vertices.items():
            dist = distance_computator.get_distance(s, state)
            heappush(dist_vertices, (dist, vertex))
        if PRM_star == 0:
            k_range = min(k, len(dist_vertices))
        else:
            k_range = k
        print(f"k: {k}")
        nearest_vertices = [
            dist_vertices[i][1] for i in range(k_range)
        ]
        return nearest_vertices

    def split_edge(self, edge_id, t):
        """Split the given edge at distance t/length where length is the length of the edge

        @return the id of the new vertex at the splitted point
        """
        edge = self.edges[edge_id][1]
        (edge1, edge2) = edge.split(t)

        self.remove_edge(edge_id)

        s = edge1.get_destination()
        # TODO: Ideally, we should check that edge1.get_destination() == edge2.get_origin()
        v = self.add_vertex(s)
        self.add_edge(edge_id[0], v, edge1)
        self.add_edge(v, edge_id[1], edge2)

        return v

    def get_vertex_path(self, root_vertex, goal_vertex):
        """Run Dijkstra's algorithm backward to compute the sequence of vertices from root_vertex to goal_vertex"""

        class ZeroCostToGoEstimator:
            """Cost to go estimator, which always returns 0."""

            def get_lower_bound(self, x):
                return 0

        Q = QueueAstar(ZeroCostToGoEstimator())
        Q.insert(goal_vertex, None, 0)
        while len(Q) > 0:
            v = Q.pop()
            if v == root_vertex:
                vertex_path = Q.get_path(goal_vertex, root_vertex)
                vertex_path.reverse()
                return vertex_path
            for u in self.parents[v]:
                edge_cost = self.edges[(u, v)][0]
                Q.insert(u, v, edge_cost)
        return []

    def get_path(self, root_vertex, goal_vertex):
        """Return a sequence of discretized states from root_vertex to goal_vertex"""
        vertex_path = self.get_vertex_path(root_vertex, goal_vertex)
        return self.get_path_from_vertex_path(vertex_path)

    def get_path_from_vertex_path(self, vertex_path):
        """Return a sequence of discretized states along the given vertex_path"""
        if len(vertex_path) == 0:
            return []

        path = []
        prev_vertex = vertex_path[0]
        for curr_ind in range(1, len(vertex_path)):
            curr_vertex = vertex_path[curr_ind]
            edge = self.edges[(prev_vertex, curr_vertex)][1]
            curr_path = edge.get_path()
            path.extend(curr_path)
            prev_vertex = curr_vertex

        return path

    def draw(self, ax):
        """Draw the graph on the axis ax"""
        for state in self.vertices.values():
            if (len(state)) == 2:
                ax.plot(state[0], state[1], "k.", linewidth=5)
            elif len(state) == 3:
                ax.plot(
                    state[0],
                    state[1],
                    marker=(3, 0, state[2] * 180 / math.pi - 90),
                    markersize=8,
                    linestyle="None",
                    markerfacecolor="black",
                    markeredgecolor="black",
                )

        for (_, edge) in self.edges.values():
            s2_ind = 1
            s1 = edge.get_discretized_state(s2_ind - 1)
            s2 = edge.get_discretized_state(s2_ind)
            while s2 is not None:
                ax.plot([s1[0], s2[0]], [s1[1], s2[1]], "k-", linewidth=1)
                s2_ind = s2_ind + 1
                s1 = s2
                s2 = edge.get_discretized_state(s2_ind)


class Tree(Graph):
    """A graph where each vertex has at most one parent"""

    def __init__(self):
        super().__init__()
        self.vertex_costs = {}  # Add a dictionary to store the cost-to-come for each vertex

    def add_edge(self, vid1, vid2, edge):
        """Add an edge from vertex with id vid1 to vertex with id vid2"""
        # Ensure that a vertex only has at most one parent (this is a tree).
        assert len(self.parents[vid2]) == 0
        super().add_edge(vid1, vid2, edge)

    def get_vertex_path(self, root_vertex, goal_vertex):
        """Trace back parents to return a path from root_vertex to goal_vertex"""
        vertex_path = [goal_vertex]
        v = goal_vertex
        while v != root_vertex:
            parents = self.parents[v]
            if len(parents) == 0:
                return []
            v = parents[0]
            vertex_path.insert(0, v)
        return vertex_path

    def remove_vertex(self, vid):
        """Remove a vertex with a given id from the graph"""

        # Remove all edges connected to the vertex
        for edge_id in list(self.edges.keys()):
            if vid in edge_id:
                self.remove_edge(edge_id)

        # Remove the vertex from the parents dictionary
        del self.parents[vid]

        # Remove the vertex from the vertices dictionary
        del self.vertices[vid]

    def get_edge_cost(self, vid1, vid2):
        """Get the cost of the edge between vertex with id vid1 and vertex with id vid2"""

        return self.edges[(vid1, vid2)][0]

    def update_edge(self, vid1, vid2, edge):
        """Update the cost and edge for an existing edge between vertex with id vid1 and vertex with id vid2"""

        self.edges[(vid1, vid2)] = (edge.get_cost(), edge)

    def set_parent(self, vid1, vid2, edge):
        """Set vid1 as the parent of vid2 in the tree"""

        # Remove the current parent of vid2, if it exists
        if len(self.parents[vid2]) > 0:
            self.remove_edge((self.parents[vid2][0], vid2))

        # Add the new edge between vid1 and vid2
        self.add_edge(vid1, vid2, edge)

    def set_vertex_cost(self, vid, cost):
        """Set the cost-to-come for the vertex with id vid"""
        self.vertex_costs[vid] = cost

    def get_vertex_cost(self, vid):
        """Set the cost-to-come for the vertex with id vid"""
        if vid not in self.vertex_costs:
            return 0.0
        return self.vertex_costs[vid]




main_rrt_star(
            cspace,
            qI,
            qG,
            edge_creator,
            distance_computator,
            collision_checker,
            obs_boundaries,
        )


def rrt_sharp(
    cspace,
    qI,
    qG,
    edge_creator,
    distance_computator,
    collision_checker,
    pG=0.1,
    numIt=100,
    tol=1e-3,
    k=10,
):
    print("inside sharp")

    def rewire(G, vs, distance_computator, edge_creator, collision_checker, k):
        qs = G.get_vertex_state(vs)
        vertices = G.get_nearest_vertices(qs, k, distance_computator)
        for vn in vertices:
            if vn != vs:
                qn = G.get_vertex_state(vn)
                (qe, edge) = stopping_configuration(
                    qs, qn, edge_creator, collision_checker, tol
                )
                if qe is not None and get_euclidean_distance(qn, qe) < tol:
                    cost_to_come = G.get_vertex_cost(vs) + edge.get_cost()
                    if cost_to_come < G.get_vertex_cost(vn):
                        G.set_parent(vs, vn, edge)
                        G.set_vertex_cost(vn, cost_to_come)

                        # Update cost-to-come of all children of vn
                        queue = [vn]
                        while queue:
                            u = queue.pop(0)
                            for v in G.parents.keys():
                                if u in G.parents[v]:
                                    cost_to_come = G.get_vertex_cost(
                                        u) + G.get_edge_cost(u, v)
                                    G.set_vertex_cost(v, cost_to_come)
                                    queue.append(v)

    G = Tree()
    root = G.add_vertex(np.array(qI))
    G.set_vertex_cost(root, 0)
    num_vertices = 1
    for i in range(numIt):
        use_goal = qG is not None and random.uniform(0, 1) <= pG
        if use_goal:
            alpha = np.array(qG)
        else:
            alpha = sample(cspace)
        vn = G.get_nearest(alpha, distance_computator, tol)
        qn = G.get_vertex_state(vn)
        (qs, edge) = stopping_configuration(
            qn, alpha, edge_creator, collision_checker, tol
        )
        if qs is None or edge is None:
            continue
        dist = get_euclidean_distance(qn, qs)
        if dist > tol:
            vs = G.add_vertex(qs)
            num_vertices += 1
            print("vs", vs)
            G.add_edge(vn, vs, edge)
            G.set_vertex_cost(vs, G.get_vertex_cost(vn) + edge.get_cost())

            # Determine k for rewiring based on the number of vertices
            k = min(k, num_vertices)
            rewire(G, vs, distance_computator,
                   edge_creator, collision_checker, k)
            if use_goal and get_euclidean_distance(qs, qG) < tol:
                return (G, root, vs)

    return (G, root, None)


# existing rewire 

def rewire(G, vs, distance_computator, edge_creator, collision_checker, radius_computer, k_nearest, eta):
        qs = G.get_vertex_state(vs)
        if k_nearest:
            k=radius_computer.get_dynamic_k_nearest_val( len(G.vertices), 0.5)
            vertices = G.get_nearest_vertices(qs, k, distance_computator)
        else: 
            radius=radius_computer.get_radius_RRT_star(len(G.vertices), eta)
            #  here eta is a user defined constant defined by user...
            vertices = G.near(alpha, radius, distance_computator)
        
        for vn in vertices:
            if vn != vs:
                qn = G.get_vertex_state(vn)
                (qe, edge) = stopping_configuration(qs, qn, edge_creator, collision_checker, tol)
                if qe is not None and (qe == qn).all() and get_euclidean_distance(qn, qe) < tol:
                    cost_to_come = G.get_vertex_cost(vs) + edge.get_cost()
                    if cost_to_come < G.get_vertex_cost(vn):
                        G.set_parent(vs, vn, edge)
                        G.set_vertex_cost(vn, cost_to_come)
                        # Update cost-to-come of all children of vn
                        queue = [vn]
                        while queue:
                            u = queue.pop(0)
                            for v in G.parents.keys():
                                if u in G.parents[v]:
                                    cost_to_come = G.get_vertex_cost(u) + G.get_edge_cost(u, v)
                                    G.set_vertex_cost(v, cost_to_come)
                                    queue.append(v)








 def rrt_star(cspace, qI, qG, edge_creator, distance_computator, collision_checker, radius_computer, k_nearest, numIt=100, tol=1e-3, eta=2.5, pG=0.1):
    """RRT* with obstacles

    @type cspace: a list of tuples (smin, smax) indicating that the C-space
        is given by the product of the tuples.
    @type qI: a tuple (x, y) indicating the initial configuration.
    @type qG: a tuple (x, y) indicating the goal configuration.
    @type edge_creator: an EdgeCreator object that includes the make_edge(s1, s2) function,
        which returns an Edge object beginning at state s1 and ending at state s2.
    @type distance_computator: a DistanceComputator object that includes the get_distance(s1, s2)
        function, which returns the distance between s1 and s2.
    @type collision_checker: a CollisionChecker object that includes the is_in_collision(s)
        function, which returns whether the state s is in collision.
    @type radius: a float indicating the search radius for near vertices.
    @type numIt: an integer indicating the maximum number of iterations.
    @type tol: a float, indicating the tolerance on the euclidean distance when checking whether
        2 states are the same

    @return (G, root, goal) where G is the tree, root is the id of the root vertex
        and goal is the id of the goal vertex (if one exists in the tree; otherwise goal will be None).
    """
    G = Tree()
    root = G.add_vertex(np.array(qI))
    G.set_vertex_cost(root, 0.0)
    goal_id = None

    for i in range(numIt):

        use_goal = qG is not None and random.uniform(0, 1) <= pG
        if use_goal:
            alpha = np.array(qG)
        else:
            alpha = sample(cspace)
        vn = G.get_nearest_vertex(alpha, distance_computator)
        qn = G.get_vertex_state(vn)
        (qs, edge) = stopping_configuration(
            qn, alpha, edge_creator, collision_checker, tol)
        if qs is None or edge is None:
            continue
        dist = get_euclidean_distance(qn, qs)
        if dist > tol:
            vs = G.add_vertex(qs)
            G.set_vertex_cost(vs, G.get_vertex_cost(vn) + edge.get_cost())
            G.add_edge(vn, vs, edge)
            radius = radius_computer.get_radius_RRT_star(len(G.vertices), eta)
            near_vertices = G.sorted_near(qs, radius, distance_computator)

            for near_id in near_vertices:
                if near_id != vs and near_id != vn:
                    q_near = G.get_vertex_state(near_id)
                    (qs_to_near, edge_to_near) = stopping_configuration(
                        qs, q_near, edge_creator, collision_checker, tol)

                    if qs_to_near is not None and edge_to_near is not None:
                        dist_near = get_euclidean_distance(qs, qs_to_near)
                        if dist_near <= tol:
                            cost_near = G.get_vertex_cost(
                                near_id) + edge_to_near.get_cost()
                            if cost_near < G.get_vertex_cost(vs):
                                G.set_parent(near_id, vs, edge_to_near)
                                G.set_vertex_cost(vs, cost_near)

            if get_euclidean_distance(qs, qG) < tol:
                if goal_id is None or G.get_vertex_cost(vs) < G.get_vertex_cost(goal_id):
                    goal_id = vs

    return (G, root, goal_id)