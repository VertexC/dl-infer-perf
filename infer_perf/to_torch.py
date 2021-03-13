import os
import time

import numpy as np
import torch

import util

def torch_runner(model_name, batch_size=1, device='gpu'):
    cuda_flag = False
    torch_device = None
    if torch.cuda.is_available() and device == 'gpu':
        print('Pytorch is using cuda!')
        cuda_flag = True
        torch_device = torch.device('cuda')

    model, shape = util.torch_model(model_name)
    model.eval()

    if cuda_flag:
        model.to(torch_device)

    data = torch.randn([batch_size] + shape, dtype=torch.float32)

    if torch.cuda.is_available():
        data = data.to(torch_device)

    def runner(data_size):
        for _ in range(data_size // batch_size):
            model(data)

    return runner


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="pytorch")
    parser.add_argument("model", help="pytorch model name")
    parser.add_argument("--device",
                        choices=['gpu', 'cpu'],
                        default='gpu',
                        help='device run')
    parser.add_argument("--batch", type=int, default=1, help='batch size')
    parser.add_argument("--size", type=int, default=256, help='data size')
    args = parser.parse_args()

    runner = torch_runner(args.model,
                          batch_size=args.batch,
                          device=args.device)

    duration = util.simple_bench(runner, args.size)
    print(duration)
