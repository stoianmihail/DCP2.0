import torch
import numpy as np
# from util import move_to_device
from disko.utils import rotation_matrix_to_euler_angles
from threading import Thread, Lock

np.set_printoptions(suppress=True)

log_mutex = Lock()

# Get the bitstring of `val`.
def get_bitstring(val):
    return "0" * (3 - len("{0:b}".format(val))) + "{0:b}".format(val)

# Invert the matrix on the set bits from `bitstring`.
def modify(X, bitstring):
    # First copy the matrix.
    # print(f'X={X}')
    Y = X.clone()

    # Apply changes.
    for index, elem in enumerate(bitstring[::-1]):
        if elem == '1':
            Y[:, 2 - index] *= -1
    return Y

# Apply SVD.
def compute_components_one(source, index=2, verbose=False):
    def apply(X, msg):
        return np.linalg.svd((X.T - np.mean(X, axis=1)).T, full_matrices=False)[:index]
    return apply(source, 'source')

# Apply SVD.
def compute_components(source, target, index=2, verbose=False):
    def apply(X, msg):
        return np.linalg.svd((X.T - np.mean(X, axis=1)).T, full_matrices=False)[:index]
    return apply(source, 'source'), apply(target, 'target')

# Apply SVD.
def torch_compute_components(source, target, index=2, verbose=False):
    def apply(X, msg):
        return torch.linalg.svd((X.T - torch.mean(X, dim=1)).T, full_matrices=False)[:index]
    return apply(source, 'source'), apply(target, 'target')

def torch_compute_components_already_centered(source, target, index=2, verbose=False):
    def apply(X, msg):
        return torch.linalg.svd(X, full_matrices=False)[:index]
    return apply(source, 'source'), apply(target, 'target')

# Check whether `R` is a valid rotation matrix.
def is_valid(R, delta=1, verbose=False):
    # TODO: change here, based on the journal!
    signs = np.array([
        [+1, -1, +1],
        [+1,  0, -1],
        [ 0, +1, +1]
    ])

    # TODO: what was this?
    couldChange = np.array([
        [0, 1, 1],
        [1, 0, 1], # TODO: in the middle should it be really 1?
        [1, 1, 0]
    ])
    eps = 1e-4
    change_eps = np.sin(np.deg2rad(delta))
    # TODO: check also if those fields can actually be zero!
    # TODO: include information about those 2 fields!
    special = [(1, 1), (2, 0)]
    def compute_sign(x):
        if x > eps:
            return +1
        if x < -eps:
            return -1
        return 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if (i, j) in special:
                continue
            sign = compute_sign(R[i][j])

            # Is 0?
            if not sign:
                continue
            if signs[i][j] != 0 and signs[i][j] != sign:
                # Sign could change easily? Then don't make the rotation invalid for this entry.
                if verbose:
                    print(f'[is_valid] try i={i}, j={j}, R={R[i][j]}')
                if torch.abs(R[i][j]) < change_eps and couldChange[i][j]:
                    continue
                if verbose:
                    print(f'[is_valid] failed at {i}, {j}: R={R[i][j]} sign={sign} actual={signs[i][j]}')
                return False
    
    # a = compute_sign(R[special[0]])
    # if a <= 0:
    #     assert 0
    #     print(f'Really! ??????????? --------------')
    #     return False
    
    return True

# from solve import solve_eq

def compute_ratio(S2, S1):
    is2D = S2[2] < 1e-3 or S1[2] < 1e-3#  torch.isclose(S2[2], torch.Tensor(0))
    if is2D:
        S1[2] = S2[2] = 1.0
    val = S2 / S1
    if is2D:
        S1[2] = S2[2] = 0.0
    return val

# TODO: read that paper with noise SVD.
def compute_sigma(S1, S2):
    sigma = 0.0
    for index in range(S1.shape[0]):
        # TODO: this is only for cube.
        sigma = max(sigma, np.abs(S1[index] - S2[index]))
    return sigma

def find_best_delta(U1, S1, U2, S2, debug=None, verbose=False):
    def rotate_axes(A, alfa):
        rad = alfa / 180 * np.pi
        v1 = A.copy()[:, 1].T.squeeze()
        v2 = A.copy()[:, 2].T.squeeze()
        v1_new = np.cos(rad) * v1 + np.sin(rad) * v2
        v2_new = -np.sin(rad) * v1 + np.cos(rad) * v2
        ret = A.copy()
        ret[:, 1] = v1_new
        ret[:, 2] = v2_new
        return ret

    
    def apply_inversions(delta):
        # if verbose:
        #     print(f'================= [START] [apply_inversions delta={delta}] ===============')
        count, bs = 0, []

        for i in range(8):
            # Modify `U2`.
            U2_new = modify(U2, get_bitstring(i))

            # Build the rotation matrix.
            
            if False:
                phi = 2
                for alfa_index in range(90 * phi):
                    alfa = alfa_index / phi
                    U2_new_rotated = rotate_axes(U2_new, alfa)

                    R = (U2_new_rotated * (compute_ratio(S2, S1))) @ U1.T
                
                    # print(f'{R}')

                    # solve_eq(U1, U2_new, compute_ratio(S2, S1))

                    if verbose:
                        print(f'R={R}')
                        print(f'angles={torch.rad2deg(rotation_matrix_to_euler_angles(torch.from_numpy(R), "zyx"))}')

                    # Is it valid?
                    if is_valid(R, delta=delta, verbose=verbose):
                        best = i
                        print(f'--- !!! --- is valid! best={best}, alfa={alfa}')
                        print(f'!!! POSSIBLE angles={torch.rad2deg(rotation_matrix_to_euler_angles(torch.from_numpy(R), "zyx"))}')
                    
                        count += 1
            else:
                R = (U2_new * (compute_ratio(S2, S1))) @ U1.T
            
                # print(f'{R}')S2

                # solve_eq(U1, U2_new, compute_ratio(S2, S1))

                # Is it valid?
                if is_valid(R, delta=delta, verbose=False):
                    if verbose:
                        # print(f'R={R}')
                        print(f'angles={rotation_matrix_to_euler_angles(R, "zyx")}')

                    bs.append(i)
                    print(f'--- !!! --- is valid! best={bs}, alfa={0}')
                    # print(f'!!! POSSIBLE angles={torch.rad2deg(rotation_matrix_to_euler_angles(torch.from_numpy(R), "zyx"))}')
                
                    count += 1
        # if verbose:
        #     print(f'================= [FINISHED] [apply_inversions delta={delta}] ===============')
        return bs

    l, r = 1., 20.
    last_best = None
    last_bs = None
    while np.abs(l - r) > 5e-2:
        mid = l + (r - l) / 2

        # print(f'l={l}, r={r}')
        
        bs = apply_inversions(mid)

        # if verbose and debug is not None:
        #     print(f'!!! rotation_ab={debug["rotation"]}')
        #     print(f'!!! euler_angles={np.rad2deg(debug["euler"])}')

        # if verbose:
        #     print(f'mid={mid}, count={count}, best={best}')

        if len(bs) == 1:
            chosen_angles = rotation_matrix_to_euler_angles(modify(U2, get_bitstring(bs[0])) * (compute_ratio(S2, S1)) @ U1.T, "zyx")
            # assert chosen_angles[0] > 1e-6 and chosen_angles[1] > 1e-6 and chosen_angles[2] > 1e-6
            # assert torch.all
            return bs[0]
        elif len(bs) > 1:
            last_bs = bs
            # print(f'best={bs}')
            # assert 0
            # last_best = best
            r = mid
        else:
            l = mid
    print(f'last_bs={last_bs}')
    for b in last_bs:
        bad = False
        for a in rotation_matrix_to_euler_angles(modify(U2, get_bitstring(b)) * (compute_ratio(S2, S1)) @ U1.T, "zyx"):
            if a < 0:
                bad = True
        if not bad:
            last_best = b  
        print(f'!!! b={b}, chosen angles={(rotation_matrix_to_euler_angles(modify(U2, get_bitstring(b)) * (compute_ratio(S2, S1)) @ U1.T, "zyx"))}')
    print(f'!!! best angles={torch.rad2deg(rotation_matrix_to_euler_angles(modify(U2, get_bitstring(last_best)) * (compute_ratio(S2, S1)) @ U1.T, "zyx"))}')
    if verbose and last_best is not None:
        print(f'!!! best angles={torch.rad2deg(rotation_matrix_to_euler_angles(modify(U2, get_bitstring(last_best)) * (compute_ratio(S2, S1)) @ U1.T, "zyx"))}')
    return last_best

def myclose(a, b, tol):
    return np.abs(a - b) < tol

def torch_solve(source, target, debug=None, verbose=False):
    if verbose:
        print(f'************************** [solve] START! ******************************')
    
    if verbose:
        print(f'source.shape={source.shape}, target.shape={target.shape}')
    
    (U1, S1), (U2, S2) = torch_compute_components(source, target, verbose=verbose)

    print(f'S1={S1}\nS2={S2}')
    # assert not torch.isclose(S1[0], S1[1], atol=1e-1) and not torch.isclose(S1[1], S1[2], atol=1e-1)

    if verbose:
        print(f'S1={S1}\nS2={S2}')

    # print(f'\n^^^^^^^^^^^^^^^^^^^^^^^^^^^\nS1={S1} {"           !!! IMPORTANT !!!" if myclose(S1[0], S1[1], tol=1e-1) or myclose(S1[1], S1[2], tol=1e-1) else ""} \nS2={S2}\n^^^^^^^^^^^^^^^^^^^^^^^^\n')

    # # TODO: singular values close?
    # # TODO: should be in function of sigma!
    # if myclose(S1[0], S1[1], tol=5e-2) or myclose(S1[1], S1[2], tol=5e-2):
    #     print(f'##################################### Use debug')
    #     log_mutex.acquire()
    #     f = open(f'debug_hyper.npy', 'wb')
    #     np.save(f, source)
    #     np.save(f, target)
    #     np.save(f, debug['rotation'][cloud_index])
    #     np.save(f, debug['euler'][cloud_index])
    #     np.save(f, debug['trans'][cloud_index])
    #     log_mutex.release()

    #     return debug['rotation'][cloud_index], debug['trans'][cloud_index], 2

    if verbose:
        print(f'S1={S1}, S2={S2}')
    best = find_best_delta(U1, S1, U2, S2, debug=debug, verbose=verbose)

    if verbose:    
        print(f'best={best}')

    # if best is None and (np.abs(S1[1] - S1[2]) < 1e-1):
    #     # U2[:,[1, 2]] = U2[:,[2, 1]]
    #     # S2[[1, 2]] = S2[[2, 1]]

    #     U2[:,[1, 2]] = U2[:,[2, 1]]
    #     S2[[1, 2]] = S2[[2, 1]]

    #     best = find_best_delta(U1, S1, U2, S2, debug=debug, verbose=verbose)
    
    # R, t, s = None, None, 0
    if best is None and (debug is not None):
        log_mutex.acquire()
        f = open(f'debug_hyper-new.npy', 'wb')
        np.save(f, source.cpu().numpy())
        np.save(f, target.cpu().numpy())
        # np.save(f, debug['rotation'])
        print(f'-->>>>>>>>>>>>>>>>>>>>>> debug={debug}')
        np.save(f, debug.cpu().numpy())
        log_mutex.release()

        assert 0

        # TODO: temporary fix.
        R = np.eye(3)
        t = np.array([0, 0, 0])
        s = 1
    # else:
    # Compute the final rotation matrix.
    assert best is not None
    R = modify(U2, get_bitstring(best)) * (compute_ratio(S2, S1)) @ U1.T

    # And translation.
    t = -R @ torch.mean(source, dim=1) + torch.mean(target, dim=1)
    return R, t

# Solve.
def torch_solve_batch(source, target, debug=None, verbose=False):
    Rs, ts, ss = [], [], []
    for index in range(source.shape[0]):
        # Solve the pair.
        R, t = torch_solve(source[index], target[index], debug=debug, verbose=verbose)

        # print(f'\n[&&&&&&&&&&&&&&&&&&&&&&&&&&]\nR.shape={R.shape}\n[&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&]\n')
        
        # Store them.
        Rs.append(R)
        ts.append(t)

    # And return.
    return torch.stack(Rs, dim=0), torch.stack(ts, dim=0)


# Solve.
# def solve_batch(source, target, debug=None):
#     Rs, ts, ss = [], [], []
#     for index in range(source.shape[0]):
#         # Solve the pair.
#         R, t, status = solve(source[index], target[index], index, debug=debug)

#         print(f'\n[&&&&&&&&&&&&&&&&&&&&&&&&&&]\nR.shape={R.shape}\n[&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&]\n')
        
#         # Store them.
#         Rs.append(torch.from_numpy(R))
#         ts.append(torch.from_numpy(t))
#         ss.append(status)

#     # And return.
#     return torch.stack(Rs, dim=0), torch.stack(ts, dim=0), ss

def main(file):
    f = open(file, 'rb')
    source = np.load(f)
    target = np.load(f)
    rotation = np.load(f)
    euler = np.load(f)
    debug = {
        'rotation' : rotation,
        'euler' : euler
    }
    R, t = solve(source, target, debug=debug, verbose=True)

    print(f'R={R}, t={t}')

import sys
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: python3 {sys.argv[0]} <debug_file>')
        sys.exit(-1)
    main(sys.argv[1])