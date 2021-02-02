import time
import json

from tf2xla import tf2xla_runner

# TODO: enlarge to proper value
# set to small for now
WARM_UP_STEPS = 5
STEPS = 10


# TODO: add single
class Task:
    def __init__(self, name, model, batch_size, params):
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.params = params

    def get_runner(self):
        if self.name == 'tf2xla':
            return tf2xla_runner(self.model, self.batch_size, **self.params)

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
    msg, valid = validate_config(config)
    if not valid:
        raise Exception("Invlida benchmark config : {}".format(msg))
    tasks = generate_tasks(config)
    benchmark = create_benchmark()
    report = []
    try:
        for task in tasks:
            runner = task.get_runner()
            duration = benchmark(task.name, runner)
            metric = "{}: {}".format(str(task), duration)
            print(metric)
            report.append(metric)
    except Exception as e:
        print(e)
    print(report)


def create_benchmark(data_size=256):
    def benchmark(name, runner, data_size=256):
        for i in range(WARM_UP_STEPS):
            runner(data_size)
        avg_time = 0
        for i in range(STEPS):
            tic = time.time()
            runner(data_size)
            toc = time.time()
            avg_time += (toc - tic)
            print("running {} round {} duration: {:.2f}".format(
                name, i, toc - tic))
        avg_time /= STEPS
        return avg_time,

    return benchmark


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="benchmark task runner")
    parser.add_argument("file", type=str, help="tf model name")

    arg = parser.parse_args()
    with open(arg.file) as f:
        data = json.load(f)

    benchmark_executor(data)
