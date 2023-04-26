from obstacle import construct_circular_obstacles, WorldBoundary2D
import numpy as np 

class radius_computer:

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
    

    def leb(self, cspace):
        # Calculate the dimension of the configuration space
        Xfree=self.compute_free_space_volume(cspace)
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


if __name__=='__main__':
    print("This is test")
    
    cspace = [(-3, 3), (-1, 1)]

    x=radius_computer()
    x.leb(cspace)