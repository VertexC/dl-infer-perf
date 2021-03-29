# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: benchmark/benchmark.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='benchmark/benchmark.proto',
  package='benchmark',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x19\x62\x65nchmark/benchmark.proto\x12\tbenchmark\"9\n\x1a\x42\x65nchmarkByteUpdateRequest\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\r\n\x05group\x18\x02 \x01(\t\"+\n\x18\x42\x65nchmarkByteUpdateReply\x12\x0f\n\x07message\x18\x01 \x01(\t2y\n\x16UpdateBenchmarkService\x12_\n\x0fUpdateBenchmark\x12%.benchmark.BenchmarkByteUpdateRequest\x1a#.benchmark.BenchmarkByteUpdateReply\"\x00\x62\x06proto3'
)




_BENCHMARKBYTEUPDATEREQUEST = _descriptor.Descriptor(
  name='BenchmarkByteUpdateRequest',
  full_name='benchmark.BenchmarkByteUpdateRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='benchmark.BenchmarkByteUpdateRequest.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='group', full_name='benchmark.BenchmarkByteUpdateRequest.group', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=97,
)


_BENCHMARKBYTEUPDATEREPLY = _descriptor.Descriptor(
  name='BenchmarkByteUpdateReply',
  full_name='benchmark.BenchmarkByteUpdateReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='benchmark.BenchmarkByteUpdateReply.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=99,
  serialized_end=142,
)

DESCRIPTOR.message_types_by_name['BenchmarkByteUpdateRequest'] = _BENCHMARKBYTEUPDATEREQUEST
DESCRIPTOR.message_types_by_name['BenchmarkByteUpdateReply'] = _BENCHMARKBYTEUPDATEREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BenchmarkByteUpdateRequest = _reflection.GeneratedProtocolMessageType('BenchmarkByteUpdateRequest', (_message.Message,), {
  'DESCRIPTOR' : _BENCHMARKBYTEUPDATEREQUEST,
  '__module__' : 'benchmark.benchmark_pb2'
  # @@protoc_insertion_point(class_scope:benchmark.BenchmarkByteUpdateRequest)
  })
_sym_db.RegisterMessage(BenchmarkByteUpdateRequest)

BenchmarkByteUpdateReply = _reflection.GeneratedProtocolMessageType('BenchmarkByteUpdateReply', (_message.Message,), {
  'DESCRIPTOR' : _BENCHMARKBYTEUPDATEREPLY,
  '__module__' : 'benchmark.benchmark_pb2'
  # @@protoc_insertion_point(class_scope:benchmark.BenchmarkByteUpdateReply)
  })
_sym_db.RegisterMessage(BenchmarkByteUpdateReply)



_UPDATEBENCHMARKSERVICE = _descriptor.ServiceDescriptor(
  name='UpdateBenchmarkService',
  full_name='benchmark.UpdateBenchmarkService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=144,
  serialized_end=265,
  methods=[
  _descriptor.MethodDescriptor(
    name='UpdateBenchmark',
    full_name='benchmark.UpdateBenchmarkService.UpdateBenchmark',
    index=0,
    containing_service=None,
    input_type=_BENCHMARKBYTEUPDATEREQUEST,
    output_type=_BENCHMARKBYTEUPDATEREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_UPDATEBENCHMARKSERVICE)

DESCRIPTOR.services_by_name['UpdateBenchmarkService'] = _UPDATEBENCHMARKSERVICE

# @@protoc_insertion_point(module_scope)
