from utils import *

if __name__ == "__main__":
    correspondence = get_correspondence()  # N images x [48 World coord, 48 image coord] 
    # correspondence = pickle.load(open("corners/correspondence.p", "rb"))
    Hs = get_homographys(correspondence)
    Hs_refined = []
    for H, (w, W) in zip(Hs, correspondence):
        Hs_refined.append(MLE_HOM_optimize(H, W, w))
    A = get_intrinsic_parameter(Hs_refined)
    k1, k2 = estimated_lens_distortion(A, Hs_refined, correspondence)
    print("Intrinsic Paratmer")
    print(A)
    print("Radial Distortion")
    print("k1", k1)
    print("k2", k2)
