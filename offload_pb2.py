# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: offload.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\roffload.proto\"+\n\x0b\x43lientInput\x12\r\n\x05image\x18\x01 \x01(\x0c\x12\r\n\x05model\x18\x02 \x01(\t\"\x1e\n\x0cServerOutput\x12\x0e\n\x06result\x18\x01 \x01(\t25\n\tOffloader\x12(\n\x07offload\x12\x0c.ClientInput\x1a\r.ServerOutput\"\x00\x62\x06proto3')



_CLIENTINPUT = DESCRIPTOR.message_types_by_name['ClientInput']
_SERVEROUTPUT = DESCRIPTOR.message_types_by_name['ServerOutput']
ClientInput = _reflection.GeneratedProtocolMessageType('ClientInput', (_message.Message,), {
  'DESCRIPTOR' : _CLIENTINPUT,
  '__module__' : 'offload_pb2'
  # @@protoc_insertion_point(class_scope:ClientInput)
  })
_sym_db.RegisterMessage(ClientInput)

ServerOutput = _reflection.GeneratedProtocolMessageType('ServerOutput', (_message.Message,), {
  'DESCRIPTOR' : _SERVEROUTPUT,
  '__module__' : 'offload_pb2'
  # @@protoc_insertion_point(class_scope:ServerOutput)
  })
_sym_db.RegisterMessage(ServerOutput)

_OFFLOADER = DESCRIPTOR.services_by_name['Offloader']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _CLIENTINPUT._serialized_start=17
  _CLIENTINPUT._serialized_end=60
  _SERVEROUTPUT._serialized_start=62
  _SERVEROUTPUT._serialized_end=92
  _OFFLOADER._serialized_start=94
  _OFFLOADER._serialized_end=147
# @@protoc_insertion_point(module_scope)
