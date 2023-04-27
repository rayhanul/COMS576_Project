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
    k_nearest=5,
):
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
            G.add_edge(vn, vs, edge)
            G.set_vertex_cost(vs, G.get_vertex_cost(vn) + dist)

            # Rewire the vertices within a certain radius
            nearest_vertices = G.get_nearest_vertices(
                qs, k_nearest, distance_computator)
            for v_near in nearest_vertices:
                q_near = G.get_vertex_state(v_near)
                (qs_rewired, edge_rewired) = stopping_configuration(
                    q_near, qs, edge_creator, collision_checker, tol
                )

                if qs_rewired is None or edge_rewired is None:
                    continue

                if np.allclose(qs_rewired, qs):
                    cost_via_near = G.get_vertex_cost(
                        v_near) + edge_rewired.get_cost()
                    if cost_via_near < G.get_vertex_cost(vs):
                        G.remove_edge((vn, vs))
                        G.add_edge(v_near, vs, edge_rewired)
                        G.set_vertex_cost(vs, cost_via_near)

            if use_goal and get_euclidean_distance(qs, qG) < tol:
                return (G, root, vs)

    return (G, root, None)


main_rrt_star(
            cspace,
            qI,
            qG,
            edge_creator,
            distance_computator,
            collision_checker,
            obs_boundaries,
        )
