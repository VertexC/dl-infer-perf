import yaml
import os

from concurrent import futures
import grpc
import pandas as pd
import proto.pkg.benchmark.benchmark_pb2_grpc as bmrpc
import proto.pkg.benchmark.benchmark_pb2 as bmpb

def update_db():
    pass


def mkdir_p(d):
    try:
        os.makedirs(d)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(d):
            pass
        else: raise

class UpdateBenchmarkService(bmrpc.UpdateBenchmarkService):
    
    def __init__(self, db):
        super(UpdateBenchmarkService, self).__init__()
        self.db = db
        
    def UpdateBenchmark(self, request, context):
        msg = db.update(request.data)
        return bmpb.BenchmarkByteUpdateReply(message='{}' % msg)

class BenchmarkDB:
    def __init__(self, url, fields):
        self.url = ''
        self.fields = ''
        print(url, fields)
        self.df = None
        self._set_up(url, fields)

    def _set_up(self, url, fields):
        print("Set up metric_db from {}".format(url))
        self.url = url
        if os.path.isfile(url):
            df = pd.read_csv(url)
        else:
            data = {k:[v] for k,v in fields.items()}
            df = pd.DataFrame.from_dict(data)
            mkdir_p(os.path.dirname(url))
            df.to_csv(url, index=False)
        self.df = df
    
    def update(self, data):
        return 'test'


def lunch_server(config_file):
    with open(config_file) as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    assert 'db_url' in content and 'fields' in content
    url = content['db_url']
    fields_list = content['fields']
    fields_dict = {}
    for fields in fields_list:
        key = fields['key']
        val = fields['default']
        fields_dict[key] = val

    bm_db = BenchmarkDB(url, fields_dict)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    bmrpc.add_UpdateBenchmarkServiceServicer_to_server(UpdateBenchmarkService(bm_db), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="benchmark server")
    parser.add_argument("config", type=str, help="server yaml config file")
    parser.add_argument("-t",
                        "--test",
                        default=256,
                        type=int,
                        help="size of test data size")
    args = parser.parse_args()

    lunch_server(args.config)