import glob
# import cv2
import numpy as np
import pickle

from constants import DATA_PATH, PATTERN_SIZE, SQUARE_SIZE, CORNER_PATH


# def get_images():
#     images = sorted(glob.glob(DATA_PATH + "/*.jpg"))
#     for image in images:
#         yield image.split("/")[-1].split(".")[0], cv2.imread(image, 0)  # greyscale


# def show_image(title, image):
#     cv2.imshow(title, image)
#     cv2.waitKey()


# def get_correspondence():
#     W_def = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), dtype=np.float64)  # World Coordinate
#     W_def[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2) * SQUARE_SIZE
#     correspondences = []
#     for image_name, image in get_images():
#         retval, corners = cv2.findChessboardCorners(image, patternSize=PATTERN_SIZE)
#         # detected
#         if retval:
#             corners = corners.reshape(-1, 2)  # image coordinmate
#             ec = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#             cv2.drawChessboardCorners(ec, PATTERN_SIZE, corners, retval)  # draw corners
#             cv2.imwrite(CORNER_PATH + "/" + image_name + ".jpg", ec)  # save to folder
#             if corners.shape[0] == W_def.shape[0]:
#                 correspondences.append([corners.astype(np.int), W_def[:, :-1].astype(np.float64)])
#     return correspondences


# DLT method requires normalized matrix
def normalize_matrix(points):
    pass


def get_homographys(correspondence):
    Hs = []
    for i in range(len(correspondence)):
        N = len(correspondence[i][0])
        A = np.zeros((2*N, 9), dtype=np.float64)
        w, W = correspondence[i]
        for j in range(len(W)):
            X, Y = W[j]
            u, v = w[j]
            A[2*j] = np.array([-X, -Y, -1, 0, 0, 0, X*u, Y*u, u])
            A[2*j+1] = np.array([0, 0, 0, -X, -Y, -1, X*v, Y*v, v])
        _, S, V = np.linalg.svd(A)
        h = V[np.argmin(S)]
        h = h / h[-1]
        H = h.reshape(3, 3)
        Hs.append(H)
    return Hs


def get_v_ij(i, j, H):
    return np.array([
        H[0, i]*H[0, j],
        H[0, i]*H[1, j] + H[1, i]*H[0, j],
        H[1, i]*H[1, j],
        H[2, i]*H[0, j] + H[0, i]*H[2, j],
        H[2, i]*H[1, j] + H[1, i]*H[2, j],
        H[2, i]*H[2, j]
    ])
    
    
def get_intrinsic_parameter(Hs):
    N = len(Hs)
    V = np.zeros((2*N, 6), np.float64)
    
    for i in range(N):
        V[2*i] = get_v_ij(0, 1, Hs[i])
        V[2*i+1] = get_v_ij(0, 0, Hs[i]) - get_v_ij(1, 1, Hs[i])
    _, s, v = np.linalg.svd(V)
    b = v[np.argmin(s)]
    b = b / b[-1]
    
    v0 = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
    lamb = b[5] - (b[3]**2 + v0*(b[1]*b[2] - b[0]*b[4]))/b[0]
    alpha = np.sqrt((lamb/b[0]))
    beta = np.sqrt(((lamb*b[0])/(b[0]*b[2] - b[1]**2)))
    gamma = -1*((b[1])*(alpha**2) *(beta/lamb))
    u0 = (gamma*v0/beta) - (b[3]*(alpha**2)/lamb)
    
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1.0]
    ])
    return A
    
    
    
def get_extrinsic(H, A):
    # Get the homography and intrinsic parameters
    inv_A = np.linalg.inv(A)
    # Get each column of the homography
    h_1 = H[:, 0]
    h_2 = H[:, 1]
    h_3 = H[:, 2]
    # Get the (average) scale factor
    lamb_1 = 1/(np.linalg.norm(np.dot(inv_A, h_1)))
    lamb_2 = 1/(np.linalg.norm(np.dot(inv_A, h_2)))
    lamb = (lamb_1 + lamb_2)/2
    # Find rotation matrix
    r_1 = lamb*np.dot(inv_A, h_1)
    r_2 = lamb*np.dot(inv_A, h_2)
    r_3 = np.cross(r_1, r_2)
    R = np.column_stack((r_1, r_2, r_3))
    # Enforce orthogonal matrix
    u,_,v = np.linalg.svd(R)
    R = np.dot(u,np.transpose(v))
    # Find translation
    t = lamb*np.dot(inv_A, h_3)
    # Find homogeneous transformation
    R_t = np.column_stack((R, t))
    return R_t
      

if __name__ == "__main__":
    # correspondence = get_correspondence()
    correspondence = pickle.load(open("corners/correspondence.p", "rb"))
    Hs = get_homographys(correspondence)
    A = get_intrinsic_parameter(Hs)
    R_t = get_extrinsic(Hs[0], A)
    print(A)
    print(R_t)
