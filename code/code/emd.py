"""随机初始化训练使用的纯 Jittor CUDA Auction EMD。

CUDA kernel 由 ``jt.code`` 即时编译，不依赖外部深度学习框架或扩展包。
为保持原训练目标的尺度，正式配置固定 1024 点、eps=0.005、50 次迭代。
"""

from __future__ import annotations

import jittor as jt


# Auction 算法维护点到点的一一分配和动态价格；这部分在 GPU 上完成竞价。
_AUCTION_HEADER = r"""
#include <cuda.h>
#include <cuda_runtime.h>

namespace jittor {

__device__ __forceinline__ float atomic_max_float(float *address, float val) {
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret)) {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old) break;
    }
    return __int_as_float(ret);
}

__global__ void auction_clear(int b, int *cnt_tmp, int *unass_cnt) {
    for (int i = threadIdx.x; i < b; i += blockDim.x) {
        cnt_tmp[i] = 0;
        unass_cnt[i] = 0;
    }
}

__global__ void auction_unassigned_count(int b, int n, const int *assignment, int *unass_cnt) {
    const int block_size = 1024;
    __shared__ int scan[block_size];
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        int point = blockIdx.y * block_size + threadIdx.x;
        scan[threadIdx.x] = point < n && assignment[i * n + point] == -1 ? 1 : 0;
        __syncthreads();
        for (int stride = 1; stride <= block_size / 2; stride *= 2) {
            int index = (threadIdx.x + 1) * stride * 2 - 1;
            if (index < block_size) scan[index] += scan[index - stride];
            __syncthreads();
        }
        if (threadIdx.x == block_size - 1) atomicAdd(&unass_cnt[i], scan[threadIdx.x]);
        __syncthreads();
    }
}

__global__ void auction_unassigned_prefix(int b, const int *unass_cnt, int *unass_sum) {
    const int block_size = 512;
    __shared__ int scan[block_size];
    scan[threadIdx.x] = threadIdx.x < b ? unass_cnt[threadIdx.x] : 0;
    __syncthreads();
    for (int stride = 1; stride <= block_size / 2; stride *= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < block_size) scan[index] += scan[index - stride];
        __syncthreads();
    }
    for (int stride = block_size / 4; stride > 0; stride /= 2) {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < block_size) scan[index + stride] += scan[index];
        __syncthreads();
    }
    if (threadIdx.x < b) unass_sum[threadIdx.x] = scan[threadIdx.x];
}

__global__ void auction_unassigned_index(
    int b, int n, const int *assignment, int *unass_idx, const int *unass_cnt,
    const int *unass_sum, int *cnt_tmp
) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        int point = blockIdx.y * 1024 + threadIdx.x;
        if (point < n && assignment[i * n + point] == -1) {
            int idx = atomicAdd(&cnt_tmp[i], 1);
            unass_idx[unass_sum[i] - unass_cnt[i] + idx] = point;
        }
    }
}

__global__ void auction_bid(
    int b, int n, const float *xyz1, const float *xyz2, float eps,
    const int *assignment, const int *assignment_inv, const float *price,
    int *bid, float *bid_inc, float *max_inc, const int *unass_cnt,
    const int *unass_sum, const int *unass_idx
) {
    const int tile = 2048;
    const int block_size = 1024;
    const int block_count = n / 1024;
    __shared__ float xyz2_buf[tile * 3];
    __shared__ float price_buf[tile];
    __shared__ float best_buf[block_size];
    __shared__ float second_buf[block_size];
    __shared__ int best_idx_buf[block_size];
    for (int batch = blockIdx.x; batch < b; batch += gridDim.x) {
        int count = unass_cnt[batch];
        if (count == 0) continue;
        int count_sum = unass_sum[batch];
        int per_block = (count + block_count - 1) / block_count;
        int threads_per_point = block_size / per_block;
        int in_block = max(min(count - (int)blockIdx.y * per_block, per_block), 0);
        float x1 = 0, y1 = 0, z1 = 0, best = -1e9f, second = -1e9f;
        int best_idx = -1, point = -1, lane = 0;
        if (threadIdx.x < threads_per_point * in_block) {
            int slot = per_block * blockIdx.y + threadIdx.x / threads_per_point + count_sum - count;
            point = unass_idx[slot];
            lane = threadIdx.x % threads_per_point;
            x1 = xyz1[(batch * n + point) * 3 + 0];
            y1 = xyz1[(batch * n + point) * 3 + 1];
            z1 = xyz1[(batch * n + point) * 3 + 2];
        }
        for (int start = 0; start < n; start += tile) {
            int end = min(n, start + tile) - start;
            for (int j = threadIdx.x; j < end * 3; j += blockDim.x)
                xyz2_buf[j] = xyz2[(batch * n + start) * 3 + j];
            for (int j = threadIdx.x; j < end; j += blockDim.x)
                price_buf[j] = price[batch * n + start + j];
            __syncthreads();
            if (point != -1) {
                int delta = (end + threads_per_point - 1) / threads_per_point;
                int left = lane * delta;
                int right = min((lane + 1) * delta, end);
                for (int k = left; k < right; ++k) {
                    float dx = xyz2_buf[k * 3 + 0] - x1;
                    float dy = xyz2_buf[k * 3 + 1] - y1;
                    float dz = xyz2_buf[k * 3 + 2] - z1;
                    float value = 3.0f - sqrtf(dx * dx + dy * dy + dz * dz) - price_buf[k];
                    if (value > best) {
                        second = best;
                        best = value;
                        best_idx = start + k;
                    } else if (value > second) {
                        second = value;
                    }
                }
            }
            __syncthreads();
        }
        best_buf[threadIdx.x] = best;
        second_buf[threadIdx.x] = second;
        best_idx_buf[threadIdx.x] = best_idx;
        __syncthreads();
        if (point != -1 && lane == 0) {
            for (int j = threadIdx.x + 1; j < threadIdx.x + threads_per_point; ++j) {
                if (best_buf[j] > best) {
                    second = max(best, second_buf[j]);
                    best = best_buf[j];
                    best_idx = best_idx_buf[j];
                } else {
                    second = max(second, best_buf[j]);
                }
            }
            float increment = best - second + eps;
            bid[batch * n + point] = best_idx;
            bid_inc[batch * n + point] = increment;
            atomic_max_float(&max_inc[batch * n + best_idx], increment);
        }
    }
}

__global__ void auction_get_max(
    int b, int n, const int *assignment, const int *bid, const float *bid_inc,
    const float *max_inc, int *max_idx
) {
    for (int batch = blockIdx.x; batch < b; batch += gridDim.x) {
        int point = threadIdx.x + blockIdx.y * blockDim.x;
        if (point < n && assignment[batch * n + point] == -1) {
            int target = bid[batch * n + point];
            float increment = bid_inc[batch * n + point];
            float maximum = max_inc[batch * n + target];
            if (increment - 1e-6f <= maximum && maximum <= increment + 1e-6f)
                max_idx[batch * n + target] = point;
        }
    }
}

__global__ void auction_assign(
    int b, int n, int *assignment, int *assignment_inv, float *price,
    const int *bid, const float *bid_inc, float *max_inc,
    const int *max_idx, bool last
) {
    for (int batch = blockIdx.x; batch < b; batch += gridDim.x) {
        int point = threadIdx.x + blockIdx.y * blockDim.x;
        if (point < n && assignment[batch * n + point] == -1) {
            int target = bid[batch * n + point];
            if (last || max_idx[batch * n + target] == point) {
                int previous = assignment_inv[batch * n + target];
                if (!last && previous != -1) assignment[batch * n + previous] = -1;
                assignment_inv[batch * n + target] = point;
                assignment[batch * n + point] = target;
                price[batch * n + target] += bid_inc[batch * n + point];
                max_inc[batch * n + target] = -1e9f;
            }
        }
    }
}

}
"""


_AUCTION_SOURCE = r"""
    @alias(xyz1, in0)
    @alias(xyz2, in1)
    @alias(assignment, out)
    int b = in0_shape0;
    int n = in0_shape1;
    int *assignment_inv = nullptr;
    int *bid = nullptr;
    int *unass_idx = nullptr;
    int *unass_cnt = nullptr;
    int *unass_sum = nullptr;
    int *cnt_tmp = nullptr;
    int *max_idx = nullptr;
    float *price = nullptr;
    float *bid_inc = nullptr;
    float *max_inc = nullptr;
    cudaMalloc(&assignment_inv, sizeof(int) * b * n);
    cudaMalloc(&bid, sizeof(int) * b * n);
    cudaMalloc(&unass_idx, sizeof(int) * b * n);
    cudaMalloc(&unass_cnt, sizeof(int) * 512);
    cudaMalloc(&unass_sum, sizeof(int) * 512);
    cudaMalloc(&cnt_tmp, sizeof(int) * 512);
    cudaMalloc(&max_idx, sizeof(int) * b * n);
    cudaMalloc(&price, sizeof(float) * b * n);
    cudaMalloc(&bid_inc, sizeof(float) * b * n);
    cudaMalloc(&max_inc, sizeof(float) * b * n);
    cudaMemset(assignment_p, 0xff, sizeof(int) * b * n);
    cudaMemset(assignment_inv, 0xff, sizeof(int) * b * n);
    cudaMemset(bid, 0, sizeof(int) * b * n);
    cudaMemset(unass_idx, 0, sizeof(int) * b * n);
    cudaMemset(unass_cnt, 0, sizeof(int) * 512);
    cudaMemset(unass_sum, 0, sizeof(int) * 512);
    cudaMemset(cnt_tmp, 0, sizeof(int) * 512);
    cudaMemset(max_idx, 0, sizeof(int) * b * n);
    cudaMemset(price, 0, sizeof(float) * b * n);
    cudaMemset(bid_inc, 0, sizeof(float) * b * n);
    cudaMemset(max_inc, 0, sizeof(float) * b * n);
    for (int iteration = 0; iteration < 50; ++iteration) {
        auction_clear<<<1, b>>>(b, cnt_tmp, unass_cnt);
        auction_unassigned_count<<<dim3(b, n / 1024, 1), 1024>>>(b, n, assignment_p, unass_cnt);
        auction_unassigned_prefix<<<1, 512>>>(b, unass_cnt, unass_sum);
        auction_unassigned_index<<<dim3(b, n / 1024, 1), 1024>>>(
            b, n, assignment_p, unass_idx, unass_cnt, unass_sum, cnt_tmp
        );
        auction_bid<<<dim3(b, n / 1024, 1), 1024>>>(
            b, n, xyz1_p, xyz2_p, 0.005f, assignment_p, assignment_inv,
            price, bid, bid_inc, max_inc, unass_cnt, unass_sum, unass_idx
        );
        auction_get_max<<<dim3(b, n / 1024, 1), 1024>>>(
            b, n, assignment_p, bid, bid_inc, max_inc, max_idx
        );
        auction_assign<<<dim3(b, n / 1024, 1), 1024>>>(
            b, n, assignment_p, assignment_inv, price, bid, bid_inc,
            max_inc, max_idx, iteration == 49
        );
    }
    cudaFree(assignment_inv);
    cudaFree(bid);
    cudaFree(unass_idx);
    cudaFree(unass_cnt);
    cudaFree(unass_sum);
    cudaFree(cnt_tmp);
    cudaFree(max_idx);
    cudaFree(price);
    cudaFree(bid_inc);
    cudaFree(max_inc);
"""


def auction_assignment(prediction, target, eps=0.005, iters=50):
    """返回每个预测点匹配到的目标点索引，反向梯度沿匹配边传播。"""
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(f"EMD shapes must match: {prediction.shape} vs {target.shape}")
    if prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError(f"EMD expects [B,N,3], got {prediction.shape}")
    if prediction.shape[0] > 512 or prediction.shape[1] != 1024:
        raise ValueError("auction EMD requires batch <= 512 and exactly 1024 points")
    if float(eps) != 0.005 or int(iters) != 50:
        raise ValueError("this audited CUDA kernel is fixed to eps=0.005 and iters=50")
    return jt.code(
        shape=(prediction.shape[0], prediction.shape[1]),
        dtype=jt.int32,
        inputs=[prediction.float32(), target.float32()],
        cuda_header=_AUCTION_HEADER,
        cuda_src=_AUCTION_SOURCE,
    )


class AuctionEMDLoss:
    """训练用 EMD：先求离散一一匹配，再返回每个 batch 的距离和。"""

    def __init__(self, eps=0.005, iters=50):
        self.eps = float(eps)
        self.iters = int(iters)

    def __call__(self, prediction, target):
        assignment = auction_assignment(prediction, target, self.eps, self.iters).stop_grad()
        batch, points, channels = target.shape
        gather_idx = assignment.reshape(batch, points, 1).broadcast((batch, points, channels))
        matched = target.stop_grad().gather(1, gather_idx)
        return ((prediction - matched) ** 2).sum(dim=-1).sum(dim=1)
