/*!
 * Copyright (c) 2022 by Microsoft Corporation.
 * Licensed under the MIT license.
 */
#include <brt/jit/compiler.h>

#include "./torch.h"

namespace brt {
namespace backend {

namespace torch {

//TCJ dynamic invoke cuda kernel
static void static_invoke(const std::vector<::torch::Tensor>& ts, const std::vector<long>& args,
                          int fd) {
  std::cout << "-----------------static_invoke-----------------" << std::endl;
  std::cout << "input Tensor shapes:" << std::endl;
  for (const auto& t : ts) {
    auto sizes = t.sizes();
    std::cout << "Tensor shape: ";
      for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << sizes[i];
        if (i < sizes.size() - 1) {
            std::cout << " x ";
        }
      }
      std::cout << std::endl;
  }

  std::cout << "Arguments:" << std::endl;
  for (const auto& arg : args) {
      std::cout << arg << " ";
  }
  std::cout << std::endl;

  std::cout << "fd:" << fd <<std::endl;

  std::vector<const void*> pargs(ts.size() + args.size()), ppargs(ts.size() + args.size()); //创建两个类型和大小相同的vector容器
  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_ON_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr();  //数据指针存储于pargs
    ppargs[i] = &pargs[i];        //所指数据的首个数据的地址
  }
  for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)args[i - ts.size()]; //pargs末尾传入arg的数据地址
    ppargs[i] = &pargs[i];                //传入首数据地址
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  jit::CUDACompiler::GetCompiler().StaticExecute(ppargs, fd, dev,
                                                 at::cuda::getDefaultCUDAStream().stream());
}

static void hetero_invoke(const std::vector<::torch::Tensor>& ts,
                          const std::vector<long>& active_blocks, int fd) {
  std::vector<const void*> pargs(ts.size() + active_blocks.size()),
      ppargs(ts.size() + active_blocks.size());

  for (int i = 0; i < (int)ts.size(); ++i) {
    CHECK_ON_CUDA(ts[i]);
    pargs[i] = ts[i].data_ptr();
    ppargs[i] = &pargs[i];
  }
  for (int i = (int)ts.size(); i < (int)pargs.size(); ++i) {
    pargs[i] = (void*)active_blocks[i - ts.size()];
    ppargs[i] = &pargs[i];
  }

  int dev = ts[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  jit::CUDACompiler::GetCompiler().HeteroExecute(ppargs, active_blocks, fd, dev,
                                                 at::cuda::getDefaultCUDAStream().stream());
}

static void homo_invoke(const std::vector<::torch::Tensor>& shared_inputs,
                        const std::vector<::torch::Tensor>& standalone_inputs,
                        const ::torch::Tensor& branch_capacities, int fd) {
  std::cout << "------------homo_invoke@brt@backend------------" << std::endl;
  for (const auto& tensor : shared_inputs) {
    std::cout << "shared_inputs Tensor shape: ";
    for (const auto& size : tensor.sizes()) {
        std::cout << size << " ";
    }
    std::cout << std::endl;
  }

  for (const auto& tensor : standalone_inputs) {
    std::cout << "standalone_inputs Tensor shape: ";
    for (const auto& size : tensor.sizes()) {
        std::cout << size << " ";
    }
    std::cout << std::endl;
  }
  
  std::cout << "branch_capacities Tensor shape: ";
    for (const auto& size : branch_capacities.sizes()) {
        std::cout << size << " ";
    }
  std::cout << std::endl;


  auto& compiler = jit::CUDACompiler::GetCompiler();
  std::vector<const void*> shared_inputs_ptr(shared_inputs.size()),
      standalone_inputs_ptr(standalone_inputs.size());
  for (int i = 0; i < (int)shared_inputs.size(); ++i) {
    CHECK_ON_CUDA(shared_inputs[i]);
    shared_inputs_ptr[i] = shared_inputs[i].data_ptr();
  }
  for (int i = 0; i < (int)standalone_inputs.size(); ++i) {
    CHECK_ON_CUDA(standalone_inputs[i]);
    standalone_inputs_ptr[i] = standalone_inputs[i].data_ptr();
  }
  int dev = shared_inputs[0].device().index();
  CHECK_EQ(0, cudaSetDevice(dev));
  auto branch_capacities_cpu = branch_capacities.to(::torch::kCPU);
  std::vector<int> branch_capacities_v(branch_capacities_cpu.data_ptr<int>(),
                                  branch_capacities_cpu.data_ptr<int>() + branch_capacities_cpu.numel());
  compiler.HomoExecute(shared_inputs_ptr, standalone_inputs_ptr, branch_capacities_v, fd, dev,
                       at::cuda::getDefaultCUDAStream().stream());
}

static std::pair<std::string, int> inject_source(const std::string& headless_code) {
  return jit::CUDACompiler::GetCompiler().InjectSource(headless_code);
}

}  // namespace torch
}  // namespace backend
}  // namespace brt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("static_invoke", &brt::backend::torch::static_invoke,
        "Generic Invoke for GPU function (CUDA)");
  m.def("hetero_invoke", &brt::backend::torch::hetero_invoke,
        "Invoke for horizontally fused GPU function (CUDA) of heterogenous kernels");
  m.def("homo_invoke", &brt::backend::torch::homo_invoke,
        "Invoke for horizontally fused GPU function (CUDA) of homogenous kernels");
  m.def("inject_source", &brt::backend::torch::inject_source, "Inject Source for GPU (CUDA)");
}