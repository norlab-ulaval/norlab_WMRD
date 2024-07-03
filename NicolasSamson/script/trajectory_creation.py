import numpy as np 
import matplotlib.pyplot as plt
from vtkmodules.numpy_interface.dataset_adapter import numpyTovtkDataArray
class eith_trajectory_generator():

    def __init__(self,r,entre_axe) -> None:
        self.r = r
        self.entre_axe = entre_axe
        self.centers = np.array([[0,entre_axe/2],[0,-entre_axe/2]])


    def calculate_defining_angle(self):

        self.defining_angle = np.arccos(2*self.r/ self.entre_axe)

    def calculate_section_point(self):
        """Calculates the 5 points that divid linear interpolation from circular interpolation.
        """
        self.defining_point = np.zeros((7,3)) # the third columns is for linear (0) or circular interpolation (1)

        l = self.r  * np.tan(self.defining_angle)

        x1 = l * np.cos(self.defining_angle)
        y1 = l * np.sin(self.defining_angle)

        self.defining_point[1,:] = np.array([x1,y1,1])
        self.defining_point[2,:] = np.array([-x1,y1,0]) # centre du cercle
        # we pass two time by 0,0 
        self.defining_point[4,:] = np.array([-x1,-y1,-1])
        self.defining_point[5,:] = np.array([x1,-y1,0]) # centre du cercle
        # We finish at the begining

        

    def interpolate_point(self,horizon=2):
        """Calculates interpolates to put one point each two meters
        """
        nb_interpolation = self.defining_point.shape[0] -1

        nb_points_final = np.array([0,0]).reshape(1,2)
        cercle_center_id = 0
        for i in range(nb_interpolation):
            start = self.defining_point[i,:2]
            end = self.defining_point[i+1,:2]
            interpolation_type = self.defining_point[i,2]

            print(f"yeah {start},{end},{interpolation_type}")
            if interpolation_type == 0:
                #1 calculate le nb de points
                norm = np.linalg.norm(end-start) 
                nb_points = int(np.ceil(norm/horizon)) 
                # calculate the multiplicators
                real_horizon = norm/nb_points
                unit_vector = (end-start)/norm
                #print(unit_vector)
                #print(start)
                #print(end)
                multiplicator = np.arange(nb_points).reshape((nb_points,1)) * real_horizon
                x_y = multiplicator * unit_vector.reshape([1,2]) + start

                

            elif interpolation_type == 1 or interpolation_type == -1:
                total_angle_of_rotation = np.pi + 2 * (np.pi/2 - self.defining_angle)
                distance_2_travel = self.r * (total_angle_of_rotation)
                
                nb_points = int(np.ceil(distance_2_travel/horizon))
                angle_increment = total_angle_of_rotation/nb_points

                real_horizon = distance_2_travel/nb_points
                
                if interpolation_type ==1:
                    start_angle = - (np.pi/2 - self.defining_angle)
                elif interpolation_type == -1:
                    start_angle = - (np.pi/2 - self.defining_angle) + np.pi
                x_y = np.zeros((nb_points,2))

                center_coordinate = self.centers[cercle_center_id,:] 
                for i in range(nb_points):
                    x_y[i,:] = np.array([np.cos(start_angle),np.sin(start_angle)]) * self.r + center_coordinate
                    start_angle += angle_increment
                
                cercle_center_id += 1 
                                
            
            nb_points_final = np.vstack((nb_points_final,x_y))

        self.x_y_trajectory = nb_points_final




    def plot_important_point(self):
        
        fig,axs = plt.subplots(1,1)

        axs.scatter(self.centers[:,0],self.centers[:,1],label="centers")
        
        axs.scatter(self.defining_point[:,0],self.defining_point[:,1],label="defining_points")
        axs.axis("equal")
        axs.legend()
        plt.show()

    def plot_trajectory(self):
        
        fig,axs = plt.subplots(2,1)

        im = axs[0].scatter(self.x_y_trajectory[:,0],self.x_y_trajectory[:,1],c=np.arange(self.x_y_trajectory.shape[0]),label="trajectory")

        axs[0].scatter(self.centers[:,0],self.centers[:,1],label="centers")
        
        axs[0].scatter(self.defining_point[:,0],self.defining_point[:,1],label="defining_points")

        axs[0].axis("equal")
        axs[0].legend()
        axs[0].set_xlabel("X position [m]")
        axs[0].set_ylabel("Y position [m]")
        axs[0].set_title("Trajectory (x,y)")
        

        fig.colorbar(im,ax=axs[0],label="Point order")

        axs[1].scatter(np.arange(self.traj_x_y_yaw.shape[0]),self.traj_x_y_yaw[:,2])
        
        axs[1].legend()
        axs[1].set_xlabel("Position number [SI]")
        axs[1].set_ylabel("Yaw angle [rad]")
        axs[1].set_title("Trajectory angle in time")
        axs[1].set_ylim(-4,4)

        
        
        plt.show()

    
    def compute_trajectory_yaw(self):

        traj_x_y_plus_1 = np.zeros((self.x_y_trajectory.shape[0]+1,self.x_y_trajectory.shape[1]))

        traj_x_y_plus_1[:-1,:] = self.x_y_trajectory

        traj_x_y_plus_1[-1,:] = self.x_y_trajectory[0,:]


        # appending the last first point at the end to 
        # calculate the angle of the last point 
        
        next_traj = traj_x_y_plus_1[1:,:]
        now_traj = traj_x_y_plus_1[0:-1,:]

        diff_traj = next_traj - now_traj

        yaw = np.arctan2(diff_traj[:,1],diff_traj[:,0])


        traj_x_y_yaw = np.zeros((self.x_y_trajectory.shape[0],self.x_y_trajectory.shape[1]+1))
        traj_x_y_yaw[:,:2] = self.x_y_trajectory
        traj_x_y_yaw[:,2] = yaw

        self.traj_x_y_yaw = traj_x_y_yaw        
        
    def export_2_vtk(self):
        self.vtk_array = numpyTovtkDataArray(self.traj_x_y_yaw)
        
    def compute_trajectory(self):

        self.calculate_defining_angle()
        self.calculate_section_point()
        self.interpolate_point(horizon=2)
        
        self.compute_trajectory_yaw()
        self.plot_trajectory()
        

traj = eith_trajectory_generator(50,100)

traj.compute_trajectory()
