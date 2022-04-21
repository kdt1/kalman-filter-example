import numpy as np
#import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

# plot preferences, interactive plotting mode
fig = plt.figure()
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()


def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx = []
    ly = []

    for i in range(len(landmarks)):
        lx.append(landmarks[i + 1][0])
        ly.append(landmarks[i + 1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    # calculate and plot covariance ellipse
    covariance = sigma[0:2, 0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    # get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:, max_ind]
    max_eigval = eigenvals[max_ind]

    # get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:, min_ind]
    min_eigval = eigenvals[min_ind]

    # chi-square value for sigma confidence interval
    chisquare_scale = 2.2789

    # calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale * max_eigval)
    height = 2 * np.sqrt(chisquare_scale * min_eigval)
    angle = np.arctan2(max_eigvec[1], max_eigvec[0])

    # generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0], estimated_pose[1]], width=width, height=height, angle=angle / np.pi * 180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo', markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',
               scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)


def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    '''your code here'''
    '''***        ***'''
    # Compute the noise free motion.
    x_new = x + delta_trans * np.cos(theta + delta_rot1)
    y_new = y + delta_trans * np.sin(theta + delta_rot1)
    theta_new = theta + delta_rot1 + delta_rot2

    # Computing the Jacobian of G with respect to the state
    G = np.array([[1.0, 0.0, -delta_trans * np.sin(theta + delta_rot1)],
                  [0.0, 1.0, delta_trans * np.cos(theta + delta_rot1)],
                  [0.0, 0.0, 1.0]])

    # Use the Motion Noise as given in the homework
    R = np.array([[0.2, 0.0, 0.0],
                  [0.0, 0.2, 0.0],
                  [0.0, 0.0, 0.02]])

    # Predict the new mean and covariance
    mu = [x_new, y_new, theta_new]
    # sigma = np.dot(np.dot(G, sigma), np.transpose(G)) + Q
    sigma = G @ sigma @ np.transpose(G) + R

    return mu, sigma


def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    # measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']


    '''your code here'''
    '''***        ***'''

    # Initialize the Jacobian of h
    H = np.array([]).reshape(0, 3)
    Z = np.array([])
    expected_Z = np.array([])

    # Vectorize measurements

    ### For each measurement , compute a row of H
    ### H[i,:]=
    for i in range(len(ids)):
        lm_id = ids[i]
        meas_range = ranges[i]
        meas_bearing = bearings[i]
        lx = landmarks[lm_id][0]
        ly = landmarks[lm_id][1]

    ######calculate expected range measurement
        q = (lx - x) ** 2 + (ly - y) ** 2
        expected_range = np.sqrt(q)

    ######calculate expected baring measurement
        expected_bearing = np.arctan((ly - y)/(lx - x)) - theta

    ######compute a row of H for each measurement
        H_i = np.array([[-(y - ly) / q, (x - lx) / q, -1],
                        [(x - lx) / expected_range, (y - ly) / expected_range, 0]])
        H = np.vstack([H, H_i])
        Z = np.hstack([Z, np.array([bearings[i], ranges[i]])])
        expected_Z = np.hstack([expected_Z, np.array([expected_bearing, expected_range])])

    # Noise covariance for the measurements
    # Gain of Kalman
    Q = 0.5 * np.eye(len(H))
    K_help = np.linalg.inv(H @ sigma @ np.transpose(H) + Q)
    K = sigma @ np.transpose(H) @ K_help

    # Kalman correction for mean and covariance
    mu = mu + np.transpose(np.dot(K, np.transpose((Z - expected_Z))))
    sigma = np.dot(np.eye(len(sigma)) - np.dot(K, H), sigma)

    return mu, sigma


def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("./Data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("./Data/sensor_data.dat")

    # initialize belief
    mu = [0.0, 0.0, 0.0]
    sigma = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    # run kalman filter
    for timestep in range(int(len(sensor_readings) / 2)):
        # plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        # perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep,'odometry'], mu, sigma)

    # perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

    plt.show('hold')


if __name__ == "__main__":
    main()
