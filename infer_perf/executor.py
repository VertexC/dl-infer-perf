import time
import json
import os, psutil
import tracemalloc
import gc

# TODO: enlarge to proper value
# set to small for now
WARM_UP_ROUNDS = 1
ROUNDS = 1
DATA_SIZE = 1


# TODO: add single
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

    def __str__(self):
        return 'name:{} model:{} batch_size:{} params:{}'.format(
            self.name, self.model, self.batch_size, self.params)


def validate_config(config):
    if 'models' not in data or len(data['models']) == 0:
        return "Missing models", False
    if 'batch_sizes' not in data or len(data['batch_sizes']) == 0:
        return "Missg batch_sizes", False
    if 'runners' not in data or len(data['runners']) == 0:
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


def benchmark_executor(config):
    process = psutil.Process(os.getpid())
    msg, valid = validate_config(config)
    if not valid:
        raise Exception("Invlida benchmark config : {}".format(msg))
    tasks = generate_tasks(config)
    report = []
    try:
        for task in tasks:
            print('Used Memory:',
                  process.memory_info().rss / 1024 / 1024, 'MB')
            runner = task.get_runner()
            if runner is None:
                continue
            duration = benchmark(task.name, runner, DATA_SIZE)
            metric = "{}: {}".format(str(task), duration)
            print(metric)
            report.append(metric)
            gc.collect()
    except Exception as e:
        print(task)
        print(e)
    print(report)
    print('Used Memory:', process.memory_info().rss / 1024 / 1024, 'MB')


def benchmark(name, runner, data_size):
    for i in range(WARM_UP_ROUNDS):
        runner(data_size)
    avg_time = 0
    for i in range(ROUNDS):
        tic = time.time()
        runner(data_size)
        toc = time.time()
        avg_time += (toc - tic)
        print("running {} round {} duration: {:.2f}".format(
            name, i, toc - tic))
    avg_time /= ROUNDS
    return avg_time


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark task runner")
    parser.add_argument("file", type=str, help="tf model name")
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

    WARM_UP_ROUNDS = args.warmup
    ROUNDS = args.rounds
    DATA_SIZE = args.size

    with open(args.file) as f:
        data = json.load(f)

    benchmark_executor(data)
