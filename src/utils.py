import glob
import cv2
import pickle
import numpy as np

from scipy import optimize

from constants import DATA_PATH, PATTERN_SIZE, SQUARE_SIZE, CORNER_PATH

def get_images():
    images = sorted(glob.glob(DATA_PATH + "/*.jpg"))
    for image in images:
        yield image.split("/")[-1].split(".")[0], cv2.imread(image, 0)  # greyscale


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey()


def get_correspondence():
    W_def = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), dtype=np.float64)  # World Coordinate
    W_def[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2) * SQUARE_SIZE
    correspondences = []
    for image_name, image in get_images():
        retval, corners = cv2.findChessboardCorners(image, patternSize=PATTERN_SIZE)
        # detected
        if retval:
            corners = corners.reshape(-1, 2)  # image coordinmate
            ec = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(ec, PATTERN_SIZE, corners, retval)  # draw corners
            cv2.imwrite(CORNER_PATH + "/" + image_name + ".jpg", ec)  # save to folder
            if corners.shape[0] == W_def.shape[0]:
                correspondences.append([W_def[:, :-1].astype(np.float64), corners.astype(np.int)])
    return correspondences


# DLT method requires normalized matrix
def normalize_matrix(points):
    pass


def get_homographys(correspondence):
    Hs = []
    for i in range(len(correspondence)):
        N = len(correspondence[i][0])
        A = np.zeros((2*N, 9), dtype=np.float64)
        W, w= correspondence[i]
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
     
     
def proj_fun(initial_h, W, w, h, N):
    W = W.reshape(N, 2)
    projected = np.zeros((2*N, ), dtype=np.float64)
    for i in range(N):
        X, Y = W[i]
        weight = h[6]*X + h[7]*Y + h[8] 
        # normalize by last element 
        projected[2*i]   = (h[0]*X + h[1]*Y + h[2]) / weight
        projected[2*i+1] = (h[3]*X + h[4]*Y + h[5]) / weight
    return projected-w
    
    
def jac_fun(initial_h, W, w, h, N):
    W = W.reshape(N, 2)
    jacobian = np.zeros((2*N, 9), np.float64)
    for i in range(N):
        x, y = W[i]
        sx = np.float64(h[0]*x + h[1]*y + h[2])
        sy = np.float64(h[3]*x + h[4]*y + h[5])
        w = np.float64(h[6]*x + h[7]*y + h[8])
        jacobian[2*i] = np.array([x/w, y/w, 1/w, 0, 0, 0, -sx*x/w**2, -sx*y/w**2, -sx/w**2])
        jacobian[2*i + 1] = np.array([0, 0, 0, x/w, y/w, 1/w, -sy*x/w**2, -sy*y/w**2, -sy/w**2])
    return jacobian


def MLE_HOM_optimize(H, W, w):
    N = len(W)
    W = W.flatten()
    w = w.flatten()
    h = H.flatten()
    
    h_prime = optimize.least_squares(fun=proj_fun,
                                     x0=h,
                                     jac=jac_fun,
                                     method="lm",
                                     args=[W, w, h, N],
                                     verbose=0)
    if h_prime.success:
        H = h_prime.x.reshape(3, 3)
        H = H/H[2, 2]
    return H
    
    
def estimated_lens_distortion(A, Hs, correspondence):
    M = len(correspondence)
    N = len(correspondence[0][0])
    D = np.zeros(((2*M*N), 2), dtype=np.float64)
    d = np.zeros(((2*M*N), 1), dtype=np.float64)
    u_c = A[0, 2]
    v_c = A[1, 2]
    for i in range(len(correspondence)):
        W, w = correspondence[i]
        R_t = get_extrinsic(Hs[i], A)
        for j in range(len(W)):
            X, Y = W[j]
            u_dot, v_dot = w[j]
            # scale = np.linalg.norm(np.array([u_dot,v_dot])).squeeze()
            # X = u_dot/scale
            # Y = v_dot/scale
            place_holder = np.matmul(R_t, np.array([[X,Y,0,1]]).T)
            x, y = place_holder[:2].squeeze()
            r = x**2+ y**2
            place_holder = np.matmul(np.matmul(A, R_t), np.array([[X, Y, 0, 1]]).T)
            u, v = place_holder[:2].squeeze()
            d_u = u - u_c
            d_v = v - v_c
            D[2*(i*len(W)+j)] = np.array([d_u*r, d_u*r**2])
            D[2*(i*len(W)+j)+1] = np.array([d_v*r, d_v*r**2])
            d[2*(i*len(W)+j)] = u_dot - u
            d[2*(i*len(W)+j)+1] = v_dot - v
    k = np.matmul(np.linalg.inv(np.matmul(D.T, D)), np.matmul(D.T, d))
    return k

if __name__ == "__main__":
    correspondence = get_correspondence()  # N images x [48 World coord, 48 image coord] 
    # correspondence = pickle.load(open("corners/correspondence.p", "rb"))
    Hs = get_homographys(correspondence)
    Hs_refined = []
    for H, (w, W) in zip(Hs, correspondence):
        Hs_refined.append(MLE_HOM_optimize(H, W, w))
    print(Hs[0] == Hs_refined[0])
    A = get_intrinsic_parameter(Hs_refined)
    k1, k2 = estimated_lens_distortion(A, Hs_refined, correspondence)
    print(k1)
    print(k2)
    print(A)
    # A_prime = get_intrinsic_parameter(Hs)

