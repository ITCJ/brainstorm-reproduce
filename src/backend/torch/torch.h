#ifndef BRT_BACKEND_TORCH_TORCH_H_
#define BRT_BACKEND_TORCH_TORCH_H_

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
// #include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#undef CHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_ON_CPU
#undef CHECK_ON_CUDA
#undef CHECK_CONTIGUOUS

#define CHECK(x) TORCH_INTERNAL_ASSERT((x), "CHECK fails.")
#define CHECK_EQ(x, y) TORCH_INTERNAL_ASSERT((x) == (y), "CHECK_EQ fails.")
#define CHECK_NE(x, y) TORCH_INTERNAL_ASSERT((x) != (y), "CHECK_NE fails.")
#define CHECK_ON_CPU(x) TORCH_INTERNAL_ASSERT(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_ON_CUDA(x) TORCH_INTERNAL_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_INTERNAL_ASSERT(x.is_contiguous(), #x " must be contiguous")

#endif  // BRT_BACKEND_TORCH_TORCH_H_
