# dl-infer-perf
A perf analysis of deep learning inference performance over pytorch/tensorflow and TensorRT/XLA/TVM.

## Environments
### TVM
docker: nvidia/cuda:11.1.1-devel-ubuntu18.0

compile tvm with llvm (clang+llvm-11.0.1-x86_64-linux-gnu-ubuntu-16.04)

**virtualenv**:
  - [tvm-torch-env](doc/tvm-torch-env-req.txt)
  - [tvm-tf-env](doc/tvm-tf-env-req.txt)
  - [tvm-onnx-env](doc/tvm-onnx-env-req.txt)

### XLA
docker: nvcr.io/nvidia/tensorflow:20.07-tf2-py3

### TensorRT
docker: nvcr.io/nvidia/tensorrt:19.09-py3 

**virtualenv**:
  - [trt-tf-env](doc/trt-tf-env-req.txt)

## Usage
### Run per optimizer&frontend
```bash
usage: executor.py [-h] [-w WARMUP] [-r ROUNDS] [-s SIZE]
                   task_file report_file

benchmark task runner

positional arguments:
  task_file             json file of tasks
  report_file           output file of results

optional arguments:
  -h, --help            show this help message and exit
  -w WARMUP, --warmup WARMUP
                        warm up rounds
  -r ROUNDS, --rounds ROUNDS
                        rounds to execute runner
  -s SIZE, --size SIZE  size of test data size
```

#### example
`python infer_perf/executor.py torch2tvm.json result.csv`

torch2tvm.json
```json
{
    "optimizer": ["tvm"],
    "fe": ["pytorch"],
    "model": ["vgg16", "resnet50", "mobilenet", "inception"],
    "batch_size": [1],
    "device": ["gpu"]
}
```
result.csv
```csv
optimizer,fe,model,batch_size,device,time
tvm,pytorch,vgg16,1,gpu,1.5274622440338135
tvm,pytorch,resnet50,1,gpu,1.535079002380371
tvm,pytorch,mobilenet,1,gpu,1.775536060333252
tvm,pytorch,inception,1,gpu,3.066736936569214
```

### Run per optimizer
```bash
bash jobs.sh <optimizer> <exp_id>
```

## Report
See [report](doc/exp/report.md) for detailed benchmark results and environment.

## Code Format
yapf infer_perf/*.py -i --style yapf.style 
