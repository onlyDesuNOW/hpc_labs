import numpy as np
import time
import pycuda.autoinit
from pycuda import driver
from pycuda.compiler import SourceModule

source_module = SourceModule("""
        __global__ void multiplicationGPU(double* A, double* B, int N, double* GPU_result){
                const int row = blockIdx.y * blockDim.y + threadIdx.y;
                const int column = blockIdx.x * blockDim.x + threadIdx.x;
                for(int i = 0; i < N; i++){
                        GPU_result[row * N + column] += A[row * N + i] * B[i * N + column];              
                }       
        }
""")


def multiplicationCPU(A, B):
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C


N = 1024
A, B = np.random.randn(N, N), np.random.randn(N, N)
GPU_result = np.zeros((N, N))

block_size = (2, 2, 1)
grid_size = (int((N + block_size[0] - 1) / 2), int((N + block_size[1] - 1) / 2))
multiplicationGPU = source_module.get_function("multiplicationGPU")

# calculation
CPU_begin = time.time()
CPU_result = multiplicationCPU(A, B)
CPU_end = time.time()

GPU_begin = time.time()
multiplicationGPU(driver.In(A), driver.In(B), driver.In(N), driver.Out(GPU_result), block=block_size, grid=grid_size)
driver.Context.synchronize()
GPU_end= time.time()


if np.allclose(GPU_result, CPU_result):
    print('Результат удовлетворитерен')
    print('Время работа на GPU: ', GPU_end - GPU_begin, '\nВремя работы на CPU: ', CPU_end - CPU_begin)
    print('Ускорени GPU результата над CPU {}'.format((CPU_end - CPU_begin) / (GPU_end - GPU_begin)))

else:
    print('Результат не сходится')
