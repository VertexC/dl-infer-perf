# Tensorflow -> TVM
[FIXME: add docker file]
- tensorflow==1.12

python infer_perf/executor.py tf_tvm.json
```bash
["name:tf2tvm model:resnet50 batch_size:1 params:{'backend': 'cuda'}: (0.01145172119140625,)", "name:tf2tvm model:resnet50 batch_size:1 params:{'backend': 'llvm'}: (0.24677371978759766,)", "name:tf2tvm model:vgg16 batch_size:1 params:{'backend': 'cuda'}: (0.008053064346313477,)", "name:tf2tvm model:vgg16 batch_size:1 params:{'backend': 'llvm'}: (0.8565731048583984,)", "name:tf2tvm model:mobilenet batch_size:1 params:{'backend': 'llvm'}: (0.037801504135131836,)", "name:tf2tvm model:inception batch_size:1 params:{'backend': 'cuda'}: (0.08290433883666992,)", "name:tf2tvm model:inception batch_size:1 params:{'backend': 'llvm'}: (0.17561697959899902,)"]
```

# Tensorflow -> XLA
sdocker run --rm --net=host --ipc=host --gpus=1 -it -v <path>:/scratch/dev/ -w /scratch/dev/ nvcr.io/nvidia/tensorflow:20.07-tf1-py3  /bin/bash

python infer_perf/executor.py xla.json
```bash
["name:tf2xla model:vgg16 batch_size:1 params:{'device': 'gpu', 'xla': False}: (2.6232311487197877,)", "name:tf2xla model:vgg16 batch_size:1 params:{'device': 'gpu', 'xla': True}: (2.592299485206604,)", "name:tf2xla model:resnet50
batch_size:1 params:{'device': 'gpu', 'xla': False}: (8.84211573600769,)", "name:tf2xla model:resnet50 batch_size:1 params:{'device': 'gpu', 'xla': True}: (11.331235599517822,)", "name:tf2xla model:mobilenet batch_size:1 params:{'device': 'gpu', 'xla': False}: (6.909182620048523,)", "name:tf2xla model:mobilenet batch_size:1 params:{'device': 'gpu', 'xla': True}: (8.754068946838379,)", "name:tf2xla model:inception batch_size:1 params:{'device': 'gpu', 'xla': False}: (12.551345491409302,)", "name:tf2xla model:inception batch_size:1 params:{'device': 'gpu', 'xla': True}: (13.299209380149842,)"]
```

# Download onnx

# Code Format
yapf infer_perf/*.py -i --style yapf.style 