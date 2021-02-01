import time

from tf2xla import tf2xla_runner

# TODO: enlarge to proper value
# set to small for now
WARM_UP_STEPS = 1
STEPS = 1


def create_runner(name, model, batch, **params):
    if name == 'tf2xla':
        return tf2xla_runner(model, batch, **params)


def executor():
    task = create_task()
    runner_infos = ['?']
    import pdb
    pdb.set_trace()
    for info in runner_infos:
        name = 'tf2xla'
        batch = 1
        model = 'vgg16'
        params = {'xla': True}
        runner = create_runner(name, model, batch, **params)
        result = task(name, runner)
        print(result)


def create_task(data_size=256):
    def task(name, runner, data_size=256):
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
        return avg_time

    return task


if __name__ == "__main__":
    executor()
