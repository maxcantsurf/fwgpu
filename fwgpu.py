from __future__ import division
from numba import jit, cuda, uint8

import numpy as np
import time


TILE_WIDTH = 32
DTYPE = uint8
UPDATE_INTERVAL = 8


def naive_fw(A, n):
    A_global_mem = cuda.to_device(A)
    
    tpb = (1, 1)
    bpg = (n, n)
    
    for k in range(n):
        naive_fw_kernel[bpg, tpb](A_global_mem, n, k)
    
    A = A_global_mem.copy_to_host()
    return A


@cuda.jit
def naive_fw_kernel(A, n, k):
    i = cuda.blockIdx.y
    j = cuda.blockIdx.x
    
    if i < n and j < n:
        s = A[i][k] + A[k][j]
        if A[i][j] > s:
            A[i][j] = s


def blocked_fw(A, n):
    
    # Stages is the number of three-staged iterations each thread will need
    # to undertake. Block size is the size of blocks the adjacency matrix is
    # broken up into
    
    t0 = time.time()
    stages = int(n/TILE_WIDTH)
    
    print('Copying memory to device...\n')
    
    # We first allocate the memory on device, then copy from host to
    # device row by row
    
    A_global_mem = cuda.device_array((n, n), dtype = np.uint8)
    
    for i in range(n):
        A_global_mem[i] = A[i]
        if (i+1)%1000 == 0:
            print(f'Copied {i+1}/{n} rows')
    
    t1 = time.time()
    
    print(f'\nCopied memory to device. Took {t1 - t0}s')
    
    block_size = (TILE_WIDTH, TILE_WIDTH)
    
    print('Starting GPU blocked Floyd-Warshall')
    print(f'Number of stages = {stages}')
    print(f'Tile width = {TILE_WIDTH}')
    print(f'Data type of A elements is {type(A[0][0])} \n')
    
    # Grid size is the number of thread blocks which are used for each stage
    
    phase_1_grid = (1, 1)
    phase_2_grid = (stages, 2)
    phase_3_grid = (stages, stages)
    
    for k in range(stages):
        base = TILE_WIDTH*k
        phase_1_kernel[phase_1_grid, block_size](A_global_mem, n, base)
        phase_2_kernel[phase_2_grid, block_size](A_global_mem, n, k, base)
        phase_3_kernel[phase_3_grid, block_size](A_global_mem, n, k, base)
        
        cuda.synchronize()
        
        if (k+1) % UPDATE_INTERVAL == 0:
            print(f'Completed stage {k+1}/{stages}')
    
    t2 = time.time()
    
    print(f'\nFinished GPU blocked Floyd-Warshall. Took {t2 - t1}s\n')
    print('Copying memory to host...')
    
    A = A_global_mem.copy_to_host()
    t3 = time.time()
    
    print(f'Copied memory to host. Took {t3 - t2}s\n')
    
    return A


@cuda.jit
def phase_1_kernel(A, n, base):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    sub_A = cuda.shared.array(shape = (TILE_WIDTH, TILE_WIDTH), dtype = DTYPE)
    sub_A[ty][tx] = A[base + ty][base + tx]
    
    cuda.syncthreads()
    
    s = 0
    for k in range(TILE_WIDTH):
        s = sub_A[ty][k] + sub_A[k][tx]
        
        if s < sub_A[ty][tx]:
            sub_A[ty][tx] = s
            
    A[base + ty][base + tx] = sub_A[ty][tx]


@cuda.jit
def phase_2_kernel(A, n, stage, base):

    if cuda.blockIdx.x != stage:
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        i_prim = base + ty
        j_prim = base + tx
        
        i = 0
        j = 0
        
        if by != 0:
            i = TILE_WIDTH*bx + ty
            j = j_prim
        else:
            j = TILE_WIDTH*bx + tx
            i = i_prim
            
        own_matrix = cuda.shared.array(shape = (TILE_WIDTH, TILE_WIDTH),
                                       dtype = DTYPE)
        prm_matrix = cuda.shared.array(shape = (TILE_WIDTH, TILE_WIDTH),
                                       dtype = DTYPE)
        
        own_matrix[ty][tx] = A[i][j]
        prm_matrix[ty][tx] = A[i_prim][j_prim]
        
        cuda.syncthreads()
        
        s = 0
        if by != 0:
            for k in range(TILE_WIDTH):
                s = own_matrix[ty][k] + prm_matrix[k][tx]
                if s < own_matrix[ty][tx]:
                    own_matrix[ty][tx] = s
        else:
            for k in range(TILE_WIDTH):
                s = prm_matrix[ty][k] + own_matrix[k][tx]
                if s < own_matrix[ty][tx]:
                    own_matrix[ty][tx] = s
                
        A[i][j] = own_matrix[ty][tx]


@cuda.jit
def phase_3_kernel(A, n, stage, base):
    
    if cuda.blockIdx.x != stage and cuda.blockIdx.y != stage:
    
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        
        i = TILE_WIDTH*by + ty
        j = TILE_WIDTH*bx + tx
        
        i_row = base + ty
        j_col = base + tx
        
        row_matrix = cuda.shared.array(shape = (TILE_WIDTH, TILE_WIDTH),
                                       dtype = DTYPE)
        col_matrix = cuda.shared.array(shape = (TILE_WIDTH, TILE_WIDTH),
                                       dtype = DTYPE)
        
        A_ij = A[i][j]
        
        row_matrix[ty][tx] = A[i_row][j]
        col_matrix[ty][tx] = A[i][j_col]
        
        cuda.syncthreads()
        
        s = 0
        for k in range(TILE_WIDTH):
            s = col_matrix[ty][k] + row_matrix[k][tx]
            if s < A_ij:
                A_ij = s

        A[i][j] = A_ij
