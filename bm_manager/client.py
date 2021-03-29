import pandas as pd
import pickle
import grpc

import proto.pkg.benchmark.benchmark_pb2 as bmpb
import proto.pkg.benchmark.benchmark_pb2_grpc as bmrpc

def lunch_client(file, server, group):
    print('Try to update benchmark {} to server {}'.format(file, server))
    df = pd.read_csv(file)
    print(df)
    df_bytes = pickle.dumps(df)

    with grpc.insecure_channel(server) as channel:
        stub = bmrpc.UpdateBenchmarkServiceStub(channel)
        response = stub.UpdateBenchmark(bmpb.BenchmarkByteUpdateRequest(data=df_bytes, group=group))
    print("Client received: {}".format(response.message))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="benchmark client")
    parser.add_argument("file", type=str, help="json file of tasks")
    parser.add_argument('server', type=str, default='localhost:50051', help='server url')
    parser.add_argument('group', type=str, default='test', help='grou name of benchmark')

    args = parser.parse_args()

    lunch_client(args.file, args.server, args.group)
