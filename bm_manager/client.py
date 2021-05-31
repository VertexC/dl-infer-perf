import pandas as pd
import pickle
import grpc

import proto.pkg.benchmark.benchmark_pb2 as bmpb
import proto.pkg.benchmark.benchmark_pb2_grpc as bmrpc


def update_df(df, server, group):
    print('Uploading df as group <{}>\n : {}'.format(group, df))
    df['group'] = pd.Series([group] * len(df), index=df.index)
    df_bytes = pickle.dumps(df)
    with grpc.insecure_channel(server) as channel:
        stub = bmrpc.UpdateBenchmarkServiceStub(channel)
        response = stub.UpdateBenchmark(
            bmpb.BenchmarkByteUpdateRequest(data=df_bytes, group=group))
    print("Client received: {}".format(response.message))


def update_file(file, server, group):
    print('Try to update benchmark {} to server {}'.format(file, server))
    df = pd.read_csv(file, index_col=False)
    update_df(df, server, group)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='update file to server')
    parser.add_argument("file", type=str, help="json file of tasks")
    parser.add_argument('--server',
                        type=str,
                        default='localhost:50051',
                        help='server url')
    parser.add_argument('--group',
                        type=str,
                        default='test',
                        help='grou name of benchmark')

    args = parser.parse_args()

    update_file(args.file, args.server, args.group)
