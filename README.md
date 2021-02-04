# dl-infer-perf
A perf analysis of deep learning inference performance over pytorch/tensorflow and TensorRT/XLA/TVM.

## Usage
[TODO: prepare proper environment]
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
### example
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
## Code Format
yapf infer_perf/*.py -i --style yapf.style 