使用mperf OpenCL相关的功能，在一些android平台上执行的时候，可能会遇到 dlopen libOpenCL.so 失败的问题, 出现报错: 
```
err: can not find opencl
err: failed to load opencl func: clGetPlatformIDs
```
work around方案是通过LD_LIBRARY_PATH环境变量找到libOpenCL.so以及libOpenCL.so间接依赖的so，进而规避vndk namespace的检查，在huawei mate40平台上实测：
```
LD_LIBRARY_PATH=/vendor/lib64/:/system/vendor/lib64/egl/ ./your_binary_run_on_mali_gpu args
```
