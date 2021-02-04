import time
import json
import gc
import util
import multiprocessing as mp
import pandas as pd
import collections


class Benchmark:
    def __init__(self, data_size=256, warmup=1, rounds=1):
        self.data_size = data_size
        self.warmup = 1
        self.rounds = 1

    def execute(self, runner):
        for i in range(self.warmup):
            runner(self.data_size)
        avg_time = 0
        for i in range(self.rounds):
            tic = time.time()
            runner(self.data_size)
            toc = time.time()
            avg_time += (toc - tic)
        avg_time /= self.rounds
        return avg_time


class Task:
    def __init__(self, name, model, batch_size, params):
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.params = params

    def get_runner(self):
        if self.name == 'tf2xla':
            from tf2xla import tf2xla_runner
            return tf2xla_runner(self.model, self.batch_size, **self.params)
        elif self.name == 'tf2tvm':
            from tf2tvm import tf2tvm_runner
            return tf2tvm_runner(self.model, self.batch_size, **self.params)
        elif self.name == 'torch2tvm':
            from torch2tvm import torch2tvm_runner
            return torch2tvm_runner(self.model, self.batch_size, **self.params)
        elif self.name == 'onnx2tvm':
            from onnx2tvm import onnx2tvm_runner
            return onnx2tvm_runner(self.model, self.batch_size, **self.params)

    def get_info(self):
        return {
            "name": self.name,
            "model": self.model,
            "batch_size": self.batch_size,
            "params": self.params,
        }

    def __str__(self):
        return 'name:{} model:{} batch_size:{} params:{}'.format(
            self.name, self.model, self.batch_size, self.params)


def validate_config(config):
    if 'models' not in config or len(config['models']) == 0:
        return "Missing models", False
    if 'batch_sizes' not in config or len(config['batch_sizes']) == 0:
        return "Missg batch_sizes", False
    if 'runners' not in config or len(config['runners']) == 0:
        return "Missing runners", False
    return "", True


def generate_tasks(config):
    tasks = []
    for model in config['models']:
        for batch_size in config['batch_sizes']:
            for runner in config['runners']:
                for name, params in runner.items():
                    tasks.append(Task(name, model, batch_size, params))
                    break
    return tasks


def execute_worker(resultq, benchmark, task):
    runner = task.get_runner()
    if runner is None:
        return
    duration = benchmark.execute(runner)
    task_info = task.get_info()
    task_info['time'] = duration
    resultq.put(task_info)


def to_report(resultq, file):
    result = collections.defaultdict(list)
    while not resultq.empty():
        for k, v in resultq.get().items():
            result[k].append(v)
    pd.DataFrame.from_dict(data=result).to_csv(file, index=False, header=True)


def execute_manager(config, file):
    msg, valid = validate_config(config)
    if not valid:
        raise Exception("Invlida benchmark config : {}".format(msg))
    tasks = generate_tasks(config)

    resultq = mp.Queue()
    benchmark = Benchmark()
    for task in tasks:
        util.memory_usage()
        try:
            p = mp.Process(target=execute_worker,
                           args=(resultq, benchmark, task))
            p.start()
            p.join()
        except Exception as e:
            print("Got exception when run {}:{}".format(task, e))
        print(resultq)

    to_report(resultq, file)
    util.memory_usage()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark task runner")
    parser.add_argument("task_file", type=str, help="json file of tasks")
    parser.add_argument("report_file", type=str, help="output file of results")
    parser.add_argument("-w",
                        "--warmup",
                        default=1,
                        type=int,
                        help="warm up rounds")
    parser.add_argument("-r",
                        "--rounds",
                        default=1,
                        type=int,
                        help="rounds to execute runner")
    parser.add_argument("-s",
                        "--size",
                        default=256,
                        type=int,
                        help="size of test data size")

    args = parser.parse_args()

    with open(args.task_file) as f:
        config = json.load(f)

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    execute_manager(config, args.report_file)
