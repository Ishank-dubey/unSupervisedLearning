import numpy as np
import matplotlib.pyplot as plt
from utils import *


# Initialize centroids
  # K is the number of clusters
  # centroids = kMeans_init_centroids(X, K)


#for iter in range(iterations):
      # Cluster assignment step:
      # Assign each data point to the closest centroid.
      # idx[i] corresponds to the index of the centroid
      # assigned to example i
#      idx = find_closest_centroids(X, centroids)

      # Move centroid step:
      # Compute means based on centroid assignments
    #  centroids = compute_centroids(X, idx, K)

#The inner-loop of the algorithm repeatedly carries out two steps:
#Assigning each training example  𝑥(𝑖)
 # to its closest centroid, and
#Recomputing the mean of each centroid using the points assigned to it.

# The K-means algorithm will always converge to some final set of means for the centroids.
# However, the converged solution may not always be ideal and depends on the initial setting of the centroids.
    # Therefore, in practice the K-means algorithm is usually run a few times with different random initializations.
    # One way to choose between these different solutions from different random initializations is to choose the one with the lowest cost function value (distortion).
#You will implement the two phases of the K-means algorithm separately
#in the next sections.
## Finding closest centroids

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)


    for i in range(X.shape[0]):
        distance = []
        for j in range(0, K):
            distance.append(np.linalg.norm(X[i] - centroids[j], axis=0))
        idx[i] = np.argmin(distance)#note only the argument


    ### START CODE HERE ###

    ### END CODE HERE ###

    return idx


X = load_data()
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    ### START CODE HERE ###
    for i in range(0, K):
        points = [];
        for j in range(0, m):
            if idx[j] == i:
                points.append(X[j])
        centroids[i] = np.mean(points, axis=0)

    ### END CODE HERE ##

    return centroids

K = 3
centroids = compute_centroids(X, idx, K)

print("The centroids are:", centroids)


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print("K-Means iteration %d/%d" % (i, max_iters - 1))

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Optionally plot progress
        #if plot_progress:
            #plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            #previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show()
    return centroids, idx
X = load_data()

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])

# Number of iterations
max_iters = 10

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)


###### Random initialization

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X

    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters

    Returns:
        centroids (ndarray): Initialized centroids
    """

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids

# Run this cell repeatedly to see different outcomes.

# Set number of centroids and max number of iterations
K = 3
max_iters = 10

# Set initial centroids by picking random examples from the dataset
initial_centroids = kMeans_init_centroids(X, K)

# Run K-Means
centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)

##### Image compression

# Load an image of a bird
original_img = plt.imread('bird_small.png')

# Visualizing the image
#plt.imshow(original_img)

print("Shape of original_img is:", original_img.shape)

#Divide by 255 so that all values are in the range 0 - 1 (not needed for PNG files)
# original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
print(X_img[0:4, :])


# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# Using the function you have implemented above.
initial_centroids = kMeans_init_centroids(X_img, K)

# Run K-Means - this can take a couple of minutes depending on K and max_iters
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)
print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])


plot_kMeans_RGB(X_img, centroids, idx, K)

# Visualize the 16 colors selected
show_centroid_colors(centroids)


#Compress the image
# Find the closest centroid of each pixel
idx = find_closest_centroids(X_img, centroids)

# Replace each pixel with the color of the closest centroid
X_recovered = centroids[idx, :]
print(X_recovered.shape)

# Reshape image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape)


# Display original image
fig, ax = plt.subplots(1,2, figsize=(16,16))
plt.axis('off')

ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
