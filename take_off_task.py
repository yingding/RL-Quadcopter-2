# from task import Task
import numpy as np
from physics_sim import PhysicsSim

class Takeoff():
    """Simple task where the goal is to lift off the ground and reach a target height.
    """
    def __init__(self, 
                 init_pose=None, 
                 init_velocities=None, 
                 init_angle_velocities=None, 
                 runtime=5., 
                 target_pos=None):
        """Initialize a TakeOff object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, 
                              init_velocities, 
                              init_angle_velocities, 
                              runtime)
                
        
        self.action_repeat = 3
        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        # self.delta_up = 0

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10., 0., 0., 0.]) 
        print("init_pos is {}".format(init_pose))
        print("target_pos is {}".format(self.target_pos))
    
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        stay_life_cons = 0.03 # keep agent stay fly
        torques_panish_delta = self.sim.pose[3:] - np.array([0,0,0])
        torques_panish = np.dot(torques_panish_delta, torques_panish_delta)
        # vertical_reward = self.sim.v[2] # y_velocity up

        reward = -.03*self.l2_norm_target() + -.002*(torques_panish) + stay_life_cons
        
        """Uses current pose of sim to return reward."""
        # reward = -.03*(abs(self.sim.pose[2] - self.target_pos[2])) + .005*self.sim.v[2]
        reward = np.clip(reward, -1, 1)
        return reward
    
    def l2_norm_target(self):
        pose_delta = self.sim.pose[:3] - self.target_pos[:3]
        return np.dot(pose_delta, pose_delta)
    
    def panish_by_grounding(self, done, reward):
        # almost grounded    
        if done and self.sim.pose[2] <= np.array([0.1])[0]:
            # crashed 
            return -50
        else:
            return reward
    def reward_by_closing(self, done, reward):
        if not done and self.l2_norm_target() <= np.array([1])[0]:
            # sphare improvement
            print("\rTargetDistanz:{}".format(self.l2_norm_target()), end="")
            return 10 - self.l2_norm_target()
        else:
            return reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds) 
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        
        next_state = np.concatenate(pose_all)
        # prevent grounding to soon
        reward = self.panish_by_grounding(done, reward)
        reward = self.reward_by_closing(done, reward)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
        

        
    
        