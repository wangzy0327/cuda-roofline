## Cuda Roofline Model

通过ncu 获取到的metrics来绘制Roofline Model，与Nsight Compute绘制的Roofline Model以及理论硬件指标进行相互印证。  

这里的测试代码以[self_gemm_template.cu](https://github.com/wangzy0327/cutlass/blob/release/4.3/examples/00_self_gemm/self_gemm_template.cu)来测试得到的数据。 详细代码见仓库 [gemm_template](https://github.com/wangzy0327/cutlass)。  

### metrics analysis

由于测试平台是在V100和A100下进行的。下面列出具体的V100和A100官方的GPUs硬件规格比较。  

| GPU Features                                  | Nvidia Tesla V100        | Nvidia Tesla A100         |
| --------------------------------------------- | ------------------------ | ------------------------- |
| GPU Artchitecture                             | Nvidia Volta             | Nvidia Ampere             |
| SMs                                           | 80                       | 108                       |
| FP32 Cores / SM                               | 64                       | 64                        |
| FP32 Cores / GPU                              | 5120                     | 6912                      |
| FP64 Cores / SM (excl. Tensor)                | 32                       | 32                        |
| FP64 Cores / GPU (excl. Tensor)               | 2560                     | 3456                      |
| INT32 Cores / SM                              | 64                       | 64                        |
| INT32 Cores / GPU                             | 5120                     | 6912                      |
| Tensor Cores / SM                             | 8                        | 4                         |
| Tensor Cores / GPU                            | 640                      | 432                       |
| Peak FP16 Tensor TFLOPS with FP16 Accumulate  | 125                      | 312/624³                  |
| Peak FP16 Tensor TFLOPS with FP32 Accumulate¹ | 125                      | 312/624³                  |
| Peak BF16 Tensor TFLOPS with FP32 Accumulate¹ | NA                       | 312/624³                  |
| Peak TF32 Tensor TFLOPS                       | NA                       | 156/312³                  |
| Peak FP64 Tensor TFLOPS                       | NA                       | 19.5                      |
| Peak INT8 Tensor TOPS¹                        | NA                       | 624/1248³                 |
| Peak INT4 Tensor TOPS¹                        | NA                       | 1248/2496³                |
| Peak FP16 TFLOPS¹(non-Tensor)                 | 31.4                     | 78                        |
| Peak BF16 TFLOPS¹(non-Tensor)                 | NA                       | 39                        |
| Peak FP32 TFLOPS¹(non-Tensor)                 | 15.7                     | 19.5                      |
| Peak FP64 TFLOPS¹ (non-Tensor)                | 7.8                      | 9.7                       |
| Peak INT32 TOPS¹,⁴                            | 15.7                     | 19.5                      |
| Memory Size                                   | 32 GB/16 GB              | 40 GB                     |
| Memory  Bandwidth                             | 900 GB/sec               | 1555 GB/sec               |
| L2 Cache Size                                 | 6144 KB                  | 40960 KB                  |
| Shared Memory Size / SM                       | Configurable up to 96 KB | Configurable up to 164 KB |



1. *Peak rates are based on GPU Boost Clock.*
2. *Four Tensor Cores in an A100 SM have 2x the raw FMA computational*
*power of eight Tensor Cores in a GV100 SM.*
3. *Effective TOPS / TFLOPS using the new Sparsity Feature*
4. *TOPS = IMAD-based integer math*



### 收集metrics

构建roofline model需要以下几个要素：

- 设备的Peak Performance(FLOPS)【可以分为Double Precision, Float Precison, Half Precision和Tensor Precision的Peak Performance】

- 设备的Peak Memory Bandwidth(Byte/s)【根据具体的场景，需要分L1 Bandwidth，L2 Bandwidth，DRAM Bandwidth】

- 模型的FLOPs【注意FLOPS(floating point operations per second)表示每秒浮点运算次数，FLOPs(floating point operations)表示浮点运算次数】

- 模型的运算时间time(s)

- 模型的memory usage(Byte)【同样需要根据具体的场景分为L1 Memory，L2 Memory，DRAM Memory】

  

**收集Peak Performance：**从`usr/local/NVIDIA-Nsight-Compute/sections`的`SpeedOfLight_HierarchicalDoubleRooflineChart.section`文件中可以看到，其对于Double Precision的Peak Performance的计算公式为

```
sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained * 2 * sm__cycles_elapsed.avg.per_second
```

同理，我们可以在`SpeedOfLight_HierarchicalSingleRooflineChart.section`中找到对于Float Precision的Peak Performance的计算公式为：

```
sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained * 2 * sm__cycles_elapsed.avg.per_second
```

在`SpeedOfLight_HierarchicalHalfRooflineChart` 和 `SpeedOfLight_HierarchicalTensorRooflineChart`中找到对于Half Precision和Tensor的Peak Performance的计算公式分别为：

```
sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained * 2 * sm__cycles_elapsed.avg.per_second
```

```
sm__inst_executed_pipe_tensor.sum.peak_sustained * 512 * sm__cycles_elapsed.avg.per_second
```

从上面提到的section文件中可以知道：

- DRAM Peak Bandwidth：
  ```dram__bytes.sum.peak_sustained * dram__cycles_elapsed.avg.per_second```
- L2 Peak Bandwidth：
  ```lts__t_bytes.sum.peak_sustained * lts__cycles_elapsed.avg.per_second```
- L1 Peak Bandwidth：
  ```l1tex__t_bytes.sum.peak_sustained * l1tex__cycles_elapsed.avg.per_second```

**收集实际的FLOPs，Bytes，Time**

- Time

  ```
  sm__cycles_elapsed.avg / sm__cycles_elapsed.avg.per_second
  ```

- FLops

  DP(Double Precision):

  ```
  sm__sass_thread_inst_executed_op_dadd_pred_on.sum + 2 x sm__sass_thread_inst_executed_op_dfma_pred_on.sum + sm__sass_thread_inst_executed_op_dmul_pred_on.sum
  ```

  SP(Single Float Precision):

  ```
  sm__sass_thread_inst_executed_op_fadd_pred_on.sum + 2 x sm__sass_thread_inst_executed_op_ffma_pred_on.sum + sm__sass_thread_inst_executed_op_fmul_pred_on.sum
  ```

  HP(Half Precision):

  ```
  sm__sass_thread_inst_executed_op_hadd_pred_on.sum + 2 x sm__sass_thread_inst_executed_op_hfma_pred_on.sum + sm__sass_thread_inst_executed_op_hmul_pred_on.sum
  ```

  Tensor Core:

  ```
  512 x sm__inst_executed_pipe_tensor.sum
  ```

- Bytes

  DRAM: ```dram__bytes.sum```

  L2:``` lts__t_bytes.sum```

  L1:``` l1tex__t_bytes.sum```

**注意：Tensor Core部分的`512`只适用于`V100` 根据GPU规格说明，个人推算 `A100` 的Tensor Core应该是1024**  



**总结收集的metrics**：在此总结我**实际**收集的metrics，其中**不包括**收集到的Peak Performance和Peak Bandwidth。  



```shell
#!/bin/bash 
 
# Time
metrics="sm__cycles_elapsed.avg,\
sm__cycles_elapsed.avg.per_second,"
 
# DP
metrics+="sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,"
 
# SP
metrics+="sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"
 
# HP
metrics+="sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,"
 
# Tensor Core
metrics+="sm__inst_executed_pipe_tensor.sum,"
 
# DRAM, L2 and L1s
metrics+="dram__bytes.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum"

ncu  --metrics $metrics --csv --target-processes all ./gpu_run > output.csv
```

最后一行的`--csv`参数是将收集到的参数以csv的格式输出，`>output.csv`是将输出保存到outpu.csv文件中，用于后续进一步处理。  

### 构建model

构建model的过程实际上就是根据公式利用已经得到的csv文件进行计算，并绘制model图片的过程。这个过程我仍然使用的是库[Roofline Model on NVIDIA GPUs](https://link.zhihu.com/?target=https%3A//gitlab.com/NERSC/roofline-on-nvidia-gpus/-/tree/roofline-hackathon-2020)中的`roofline.py`和`postprocess.py`两个文件

其中`roofline.py`就是根据输入的参数绘制model图片的函数。

而`postprocess.py`是处理csv文件，并调用`roofline.py`中函数的程序。具体的使用方法可以参考库中的README.md文件。

构建[roofline.py](src/roofline.py)是将设备的Peak Bandwidth和Peak Performance写死了的，因此需要将roofline.py中的cmpRoofs和memRoofs数组修改为自己设备的参数。