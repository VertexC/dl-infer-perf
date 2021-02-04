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
    def __init__(self, fe, optimizer, model, batch_size, device):
        self.fe = fe
        self.optimizer = optimizer
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.params = {}

    def get_runner(self):
        if self.optimizer == 'xla':
            from to_xla import xla_runner
            return xla_runner(self.fe,
                              self.model,
                              self.batch_size,
                              self.device,
                              xla=True)
        elif self.optimizer == 'tvm':
            from to_tvm import tvm_runner
            return tvm_runner(self.fe, self.model, self.batch_size,
                              self.device)
        else:
            return None

    def get_info(self):
        return {
            "optimizer": self.optimizer,
            "fe": self.fe,
            "model": self.model,
            "batch_size": self.batch_size,
            "device": self.device,
        }

    def __str__(self):
        return 'name:{} model:{} batch_size:{} params:{}'.format(
            self.name, self.model, self.batch_size, self.params)


def validate_config(config):
    if 'fe' not in config or len(config['fe']) == 0:
        return 'Missing frontend', False
    if 'optimizer' not in config:
        config['optimizer'] = ''
    if 'model' not in config or len(config['model']) == 0:
        return 'Missing models', False
    if 'batch_size' not in config or len(config['batch_size']) == 0:
        return "Missg batch_sizes", False
    return '', True


def generate_tasks(config):
    tasks = []
    for model in config['model']:
        for batch_size in config['batch_size']:
            for optimizer in config['optimizer']:
                for device in config['device']:
                    for fe in config['fe']:
                        tasks.append(
                            Task(fe, optimizer, model, batch_size, device))
    return tasks


def execute_worker(resultq, benchmark, task):
    runner = task.get_runner()
    task_info = task.get_info()
    if runner is None:
        print("Get invalid task: {}".format(task_info))
        return
    print("Star to run stask: {}".format(task_info))
    duration = benchmark.execute(runner)
    task_info['time'] = duration
    print(task_info)
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
        try:
            p = mp.Process(target=execute_worker,
                           args=(resultq, benchmark, task))
            p.start()
            p.join()
        except Exception as e:
            print("Got exception when run {}:{}".format(task, e))
    to_report(resultq, file)


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
