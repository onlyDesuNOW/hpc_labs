import numpy as np
import pycuda.autoinit
from pycuda import driver
from pycuda.compiler import SourceModule
import time

source_module = SourceModule("""
                __global__ void piCalc(double *x, double *y, int N, double *count) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        int threadCount = gridDim.x * blockDim.x;
        int count_gpu = 0;
        for (int i = idx; i < N; i += threadCount) {
                if (x[i] * x[i] + y[i] * y[i] < 1) {
                        count_gpu++;
                }
        }
        atomicAdd(count , count_gpu);
}
""")


def calc_cpu(X, Y, N):
    count_cpu = 0
    for i in range(N):
        if X[i] ** 2 + Y[i] ** 2 < 1:
            count_cpu += 1
    return count_cpu * 4 / N


N = 65536 * 16
X, Y = np.random.random(N), np.random.random(N)

block_size = (256, 1, 1)
grid_size = (int(N / (128 * block_size[0])), 1)

count = np.zeros(1)
piCalc = source_module.get_function("piCalc")

# calculation
CPU_begin = time.time()
pi_cpu = calc_cpu(N)
CPU_end = time.time()

GPU_begin = time.time()
piCalc(driver.In(X), driver.In(Y), driver.In(N), driver.Out(count), block=block_size, grid=grid_size)
driver.Context.synchronize()
GPU_end = time.time()

print('Время работа на GPU: ', GPU_end - GPU_begin, '\nВремя работы на CPU: ', CPU_end - CPU_begin)
print('Result GPU ', count * 4 / N, '\nResult CPU ', pi_cpu)
print('Ускорени GPU результата над CPU {}'.format((CPU_end - CPU_begin) / (GPU_end - GPU_begin)))
