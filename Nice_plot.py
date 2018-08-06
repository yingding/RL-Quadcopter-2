import matplotlib.pyplot as plt
#matplotlib inline
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.linalg import expm
import copy
import glob
import os, sys, subprocess

# Patch to 3d axis to remove margins around x, y and z limits.
# Taken from here: https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
###patch end###


# Rotation using euler angles from here:
# https://gist.github.com/machinaut/29d0e21b544b4a36082c761c439144d6
def rotateByEuler(points, xyz):
    ''' Rotate vector v (or array of vectors) by the euler angles xyz '''
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    for theta, axis in zip(xyz, np.eye(3)):
        points = np.dot(np.array(points), expm(np.cross(np.eye(3), axis*-theta)).T)
    return points


def plot(results, fancy=True):
    my_plt = get_plot(results, fancy=fancy)
    my_plt.show()
    
def generate_video_for_one_episode(epi_id, epi_result, fancy=False, plotDims=None, skipRate=10, dirpath=None):
    """
    generate video for a particular episode of learning or simulation
    """
    if dirpath is None:
        dirpath = "./videos/frames/video{:04}_episode{:04}/".format(epi_id, i)
    frame = 0
    
    print("Time t has length {}".format(len(epi_result['time'])))
    for i in range(0, len(epi_result['time']), skipRate):
        filepath = "{}frame{:04}.png".format(dirpath, frame)
        # print("Epi: {0:2d} frame {1:3d}".format(epi_id, frame))
        print("\rGenerating image {}".format(filepath), end="")
        # print("Generating image {}".format(filepath), end="\r")
        sys.stdout.flush()
        
        # Create image
        my_plt = get_plot(epi_result, fancy=fancy, plotDims=plotDims, framesMax=i)
        my_plt.savefig(filepath)
        # close the current after the plot.
        my_plt.close()
        frame += 1;
    
    video_name = "{}video_episode_{}.mp4".format(dirpath,epi_id)
    # https://www.ostechnix.com/20-ffmpeg-commands-beginners/
    # -r – Set the frame rate. I.e the number of frames to be extracted into images per second. The default value is 25.
    # -r 30 output framerate is 30
    # https://video.stackexchange.com/questions/13066/how-to-encode-a-video-at-30-fps-from-images-taken-at-7-fps
    # -framerate 8 input frame rate is 8
    
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', "{}frame%04d.png".format(dirpath), '-r', '30', '-pix_fmt', 'yuv420p',
        video_name
    ])
    print("")
    print("Created video {}".format(video_name))
    
    for file_name in glob.glob("{}*.png".format(dirpath)):
        os.remove(file_name)
    print("cleaned all image file in folder {}".format(dirpath))
      

def get_plot(results_origin, fancy=True, plotDims=None, framesMax=None):
    # pre-adjust the results with framesMax
    if not (framesMax is None):
        n = len(results_origin['x'])
        # framesMax is the index begins with 0
        if framesMax < n - 1:
            # deep copy
            results = copy.copy(results_origin)
            for key, value in results.items():
                results[key] = value[:(framesMax-n+1)]
        else:
            # point assignment
            results = results_origin
    else:
        # point assignment
        results = results_origin
            

    # Set up axes grid. ###############################################################
    fig = plt.figure(figsize=(20,15))
    ax1 = plt.subplot2grid((20, 40), (0, 0), colspan=24, rowspan=20, projection='3d')
    ax2 = plt.subplot2grid((20, 40), (1, 28), colspan=12, rowspan=4)
    ax3 = plt.subplot2grid((20, 40), (6, 28), colspan=12, rowspan=4)
    ax4 = plt.subplot2grid((20, 40), (11, 28), colspan=12, rowspan=4)
    ax5 = plt.subplot2grid((20, 40), (15, 28), colspan=12, rowspan=4)


    # Plot 3d trajectory and copter. ##################################################
    c = 0.0
    plt.rcParams['grid.color'] = [c, c, c, 0.075]
    mpl.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.xmargin'] = 0


    if plotDims and plotDims[0]:
        plotLimitXY = plotDims[0]
    else:
        plotLimitXY = 14
    if plotDims and plotDims[1]:
        plotLimitZ = plotDims[1]
    else:
        plotLimitZ = 30

    quadSize = 0.5
    nPointsRotor = 15
    pointsQuadInitial = [[-quadSize, -quadSize, 0], [-quadSize, quadSize, 0], [quadSize, quadSize, 0], [quadSize, -quadSize, 0]]
    pointsRotorInitial = np.vstack(( np.sin(np.linspace(0., 2.*np.pi, nPointsRotor)),
                                     np.cos(np.linspace(0., 2.*np.pi, nPointsRotor)),
                                     np.repeat(0.0, nPointsRotor))).T * quadSize * 0.8

    # Create 3d plot.

    #ax = fig.gca(projection='3d')
    ax1.view_init(12, -55)
    ax1.dist = 7.6

    # Plot trajectories projected.
    xLimited = [x for x in results['x'] if np.abs(x) <= plotLimitXY]
    yLimited = [y for y in results['y'] if np.abs(y) <= plotLimitXY]
    zLimited = [z for z in results['z'] if z <= plotLimitZ]
    
    l = min(len(xLimited), len(yLimited))
    ax1.plot(xLimited[0:l], yLimited[0:l], np.repeat(0.0, l), c='darkgray', linewidth=0.9)
    l = min(len(xLimited), len(zLimited))
    ax1.plot(xLimited[0:l], np.repeat(plotLimitXY, l), zLimited[0:l], c='darkgray', linewidth=0.9)
    l = min(len(yLimited), len(zLimited))
    ax1.plot(np.repeat(-plotLimitXY, l), yLimited[0:l], zLimited[0:l], c='darkgray', linewidth=0.9)

    # Plot trajectory 3d.
    ax1.plot(results['x'], results['y'], results['z'], c='gray', linewidth=0.5)

    # Plot copter.
    nTimesteps = len(results['x'])
    # Colors from here: https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    colors = np.array([ [230, 25, 75, 255],
                        [60, 180, 75, 255],
                        [255, 225, 25, 255],
                        [0, 130, 200, 255]]) / 255.




    for t in range(nTimesteps):
        # Plot copter position as dot on trajectory for each full second. ******
        if results['time'][t]%1.0 <= 0.025 or results['time'][t]%1.0 >= 0.975:
            ax1.scatter([results['x'][t]], [results['y'][t]], [results['z'][t]], s=5, c=[0., 0., 0., 0.3])
        alpha1 = 0.96*np.power(t/nTimesteps, 20)+0.04
        alpha2 = 0.5 * alpha1
        # Plot frame. **********************************************************
        if fancy or t == nTimesteps -1:
            # Rotate frame it.
            pointsQuad = rotateByEuler(pointsQuadInitial, np.array([results['phi'][t], results['theta'][t], results['psi'][t]]))
            # Move it.
            pointsQuad += np.array([results['x'][t], results['y'][t], results['z'][t]])
        # Plot frame projections for last time step.
        if t == nTimesteps -1:
            # Z plane.
            if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['y'][t]) <= plotLimitXY:
                ax1.plot(pointsQuad[[0,2], 0], pointsQuad[[0,2], 1], [0., 0.], c=[0., 0., 0., 0.1])
                ax1.plot(pointsQuad[[1,3], 0], pointsQuad[[1,3], 1], [0., 0.], c=[0., 0., 0., 0.1])
            # Y plane.
            if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                ax1.plot(pointsQuad[[0,2], 0], [plotLimitXY, plotLimitXY], pointsQuad[[0,2], 2], c=[0., 0., 0., 0.1])
                ax1.plot(pointsQuad[[1,3], 0], [plotLimitXY, plotLimitXY], pointsQuad[[1,3], 2], c=[0., 0., 0., 0.1])
            # X plane.
            if np.abs(results['y'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                ax1.plot([-plotLimitXY, -plotLimitXY], pointsQuad[[0,2], 1], pointsQuad[[0,2], 2], c=[0., 0., 0., 0.1])
                ax1.plot([-plotLimitXY, -plotLimitXY], pointsQuad[[1,3], 1], pointsQuad[[1,3], 2], c=[0., 0., 0., 0.1])
        # Plot frame for all other time steps.
        if fancy:
            ax1.plot(pointsQuad[[0,2], 0], pointsQuad[[0,2], 1], pointsQuad[[0,2], 2], c=[0., 0., 0., alpha2])
            ax1.plot(pointsQuad[[1,3], 0], pointsQuad[[1,3], 1], pointsQuad[[1,3], 2], c=[0., 0., 0., alpha2])

        # Plot rotors. *********************************************************
        # Rotate rotor.
        if fancy or t == nTimesteps -1:
            pointsRotor = rotateByEuler(pointsRotorInitial, np.array([results['phi'][t], results['theta'][t], results['psi'][t]]))
        # Move rotor for each frame point.
        for i, color in zip(range(4), colors):
            if fancy or t == nTimesteps -1:
                pointsRotorMoved = pointsRotor + pointsQuad[i]
            # Plot rotor projections.
            if t == nTimesteps -1:
                # Z plane.
                if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['y'][t]) <= plotLimitXY:
                    ax1.add_collection3d(Poly3DCollection([list(zip(pointsRotorMoved[:,0], pointsRotorMoved[:,1], np.repeat(0, nPointsRotor)))], facecolor=[0.0, 0.0, 0.0, 0.1]))
                # Y plane.
                if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                    ax1.add_collection3d(Poly3DCollection([list(zip(pointsRotorMoved[:,0], np.repeat(plotLimitXY, nPointsRotor), pointsRotorMoved[:,2]))], facecolor=[0.0, 0.0, 0.0, 0.1]))
                # X plane.
                if np.abs(results['y'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                    ax1.add_collection3d(Poly3DCollection([list(zip(np.repeat(-plotLimitXY, nPointsRotor), pointsRotorMoved[:,1], pointsRotorMoved[:,2]))], facecolor=[0.0, 0.0, 0.0, 0.1]))
            # Outline.
            if t == nTimesteps-1:
                ax1.plot(pointsRotorMoved[:,0], pointsRotorMoved[:,1], pointsRotorMoved[:,2], c=color[0:3].tolist()+[alpha1], label='Rotor {:g}'.format(i+1))
            elif fancy:
                ax1.plot(pointsRotorMoved[:,0], pointsRotorMoved[:,1], pointsRotorMoved[:,2], c=color[0:3].tolist()+[alpha1])
            # Fill.
            if fancy or t == nTimesteps -1:
                ax1.add_collection3d(Poly3DCollection([list(zip(pointsRotorMoved[:,0], pointsRotorMoved[:,1], pointsRotorMoved[:,2]))], facecolor=color[0:3].tolist()+[alpha2]))



    ax1.legend(bbox_to_anchor=(0.0 ,0.0 , 0.95, 0.85), loc='upper right')
    c = 'r'
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_xlim(-plotLimitXY, plotLimitXY)
    ax1.set_ylim(-plotLimitXY, plotLimitXY)
    ax1.set_zlim(0, plotLimitZ)
    ax1.set_xticks(np.arange(-plotLimitXY, plotLimitXY+2, 2))
    ax1.set_yticks(np.arange(-plotLimitXY, plotLimitXY+2, 2))
    ax1.set_zticks(np.arange(0, plotLimitZ+2, 2))
    ax1.set_title("3D Plot")

    '''
    # Plot rotor speeds.
    ax2.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second', c=colors[0])
    ax2.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second', c=colors[1])
    ax2.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second', c=colors[2])
    ax2.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second', c=colors[3])
    ax2.set_ylim(0, 1000)
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('f [Hz]')
    '''

    # Plot copter angles.
    ax2.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax2.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['phi']], label='$\\alpha_x$')
    ax2.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['theta']], label='$\\alpha_y$')
    ax2.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['psi']], label='$\\alpha_z$')
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('$\\alpha$ [rad]')
    ax2.legend()
    ax2.set_title('Angles')

    # Plot copter velocities.
    ax3.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax3.plot(results['time'], results['x_velocity'], label='$V_x$')
    ax3.plot(results['time'], results['y_velocity'], label='$V_y$')
    ax3.plot(results['time'], results['z_velocity'], label='$V_z$')
    ax3.set_ylim(-20, 20)
    ax3.set_xlabel('t [s]')
    ax3.set_ylabel('V [$m\,s^{1}$]')
    ax3.legend()
    ax3.set_title('Velocities')


    # Plot copter turn rates.
    ax4.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax4.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    ax4.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    ax4.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    ax4.set_ylim(-3, 3)
    ax4.set_xlabel('t [s]')
    ax4.set_ylabel('$\omega$ [$rad\,s^{1}$]')
    ax4.legend()

    # Plot reward.
    ax5.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax5.plot(results['time'], results['reward'], label='Reward')
    #ax5.set_ylim(-10, 10)
    ax5.set_xlabel('t [s]')
    ax5.set_ylabel('Reward')
    #ax5.legend()

    return plt
    # Done :)
    # plt.show()