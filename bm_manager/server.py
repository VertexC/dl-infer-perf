import yaml
import os
import pickle
import threading

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

'''
df: pd.dataframe
fields: {column_name: value}

return: the index of df satisfy match fields, -1 if not found
'''
def query(df, fields):
    if len(fields) == 0:
        return df.index
    else:
        k,v = fields[0]
        df = df[df[k] == v]
        if len(df) == 0:
            return None
        else:
            return query(df, fields[1:])


class UpdateBenchmarkService(bmrpc.UpdateBenchmarkService):
    def __init__(self, db):
        super(UpdateBenchmarkService, self).__init__()
        self.db = db
        
    def UpdateBenchmark(self, request, context):
        print('Receive UpdateBenchmark request')
        df = pickle.loads(request.data)
        print('Server get df\n:{}'.format(df))
        msg = self.db.update(df)
        if msg:
            message = 'Failed to udpate: {}'.format(msg)
        else:
            message = 'Update Success'
        return bmpb.BenchmarkByteUpdateReply(message=message)

class BenchmarkDB:
    def __init__(self, url, key_f, val_f):
        self.url = url
        self.key_f = key_f
        self.val_f = val_f
        self._df = None
        self._set_up(url, key_f | val_f)
        self._lock = threading.Lock()

    def _set_up(self, url, fields):
        print("Set up metric_db from {}".format(url))
        if os.path.isfile(url):
            df = pd.read_csv(url)
        else:
            data = {k:None for k in fields}
            df = pd.DataFrame.from_dict(data)
            mkdir_p(os.path.dirname(url))
            df.to_csv(url, index=False)
        self._df = df
    
    def update(self, df):
        print('Checking format...')
        import pdb; pdb.set_trace()
        headers = df.head()
        if set(headers) != set(self.key_f | self.val_f):
            return 'headers not match, expect {}, got {}'.format(self.key_f | self.val_f, headers)
        self._lock.acquire()
        to_drop = []
        for _, r in df.iterrows():
            row = [(k, r[k]) for k in self.key_f]
            index = query(self._df, row)
            if index is not None:
                to_drop += index.tolist()
        self._df = self._df.drop(to_drop)
        self._df = pd.concat([self._df, df], ignore_index=True)
        self._lock.release()
        print('Data updated:\n {}'.format(self._df))
        return None

def lunch_server(config_file):
    with open(config_file) as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    url = content['db_url']
    key_list = content['key_fields']
    val_list = content['val_fields']
    key_f = set(key_list.split(','))
    val_f = set(val_list.split(','))
    bm_db = BenchmarkDB(url, key_f, val_f)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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