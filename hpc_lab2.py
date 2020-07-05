import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
import numpy as np
import cv2
import timeit

def bilateral(img, sigma_d, sigma_r):
    n_img = np.zeros(img.shape)
    w = img.shape[0]
    h = img.shape[1]
    for i in range(1, w-1):
        for j in range(1, h-1):
            n_img[i, j] = pixel(img, i, j, sigma_d, sigma_r)
    return n_img

def pixel(img, i, j, sigma_d, sigma_r):
    c = 0
    s = 0
    for k in range(i-1, i+2):
        for l in range(j-1, j+2):
            g = np.exp(-((k - i) ** 2 + (l - j) ** 2) / sigma_d ** 2)
            i1 = img[k, l] / 255
            i2 = img[i, j] / 255
            r = np.exp(-((i1 - i2)*255) ** 2 / sigma_r ** 2)
            c += g*r
            s += g * r * img[k, l]
    result = s / c
    return result


mod = compiler.SourceModule("""
texture<unsigned int, 2, cudaReadModeElementType> tex;

__global__ void bilateral(unsigned int * __restrict__ d_result, const int M, const int N, const float sigma_d, const float sigma_r)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;


    if ((i<M)&&(j<N)) {
        float s = 0;
        float c = 0;
        for (int l = i-1; l <= i+1; l++){
            for (int k = j-1; k <= j+1; k++){
                float img1 = tex2D(tex, k, l)/255;
                float img2 = tex2D(tex, i, j)/255;
                float g = exp(-(pow(k - i, 2) + pow(l - j, 2)) / pow(sigma_d, 2));
                float r = exp(-pow((img1 - img2)*255, 2) / pow(sigma_r, 2));
                c += g*r;
                s += g*r*tex2D(tex, k, l);
            }
        }
        d_result[i*N + j] = s / c;
    }


}
""")
bilateral_interpolation = mod.get_function("bilateral")

IMG = 'input.bmp'
img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
M, N = img.shape

sigma_d = 100
sigma_r = 10
block = (16, 16, 1)
grid = (int(np.ceil(M/block[0])),int(np.ceil(N/block[1])))
start = driver.Event()
stop = driver.Event()

start.record()
tex = mod.get_texref("tex")
tex.set_filter_mode(driver.filter_mode.LINEAR)
tex.set_address_mode(1, driver.address_mode.MIRROR)
driver.matrix_to_texref(img.astype("int32"), tex, order="C")
gpu_result = np.zeros((M, N), dtype=np.uint32)
bilateral_interpolation(driver.Out(gpu_result), np.int32(M), np.int32(N), np.float32(sigma_d), np.float32(sigma_r), block=block, grid=grid, texrefs=[tex])
stop.record()
stop.synchronize()
gpu_time = stop.time_since(start)
print(gpu_time)
cv2.imwrite("output.bmp", gpu_result.astype("int8"))


start = timeit.default_timer()
cpu_result = bilateral(img, sigma_d, sigma_r)
cpu_time = timeit.default_timer() - start
print(cpu_time * 1e3)
