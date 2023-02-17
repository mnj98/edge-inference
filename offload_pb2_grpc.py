# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import offload_pb2 as offload__pb2


class OffloaderStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.offload = channel.unary_unary(
                '/Offloader/offload',
                request_serializer=offload__pb2.ClientInput.SerializeToString,
                response_deserializer=offload__pb2.ServerOutput.FromString,
                )


class OffloaderServicer(object):
    """Missing associated documentation comment in .proto file."""

    def offload(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_OffloaderServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'offload': grpc.unary_unary_rpc_method_handler(
                    servicer.offload,
                    request_deserializer=offload__pb2.ClientInput.FromString,
                    response_serializer=offload__pb2.ServerOutput.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Offloader', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Offloader(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def offload(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Offloader/offload',
            offload__pb2.ClientInput.SerializeToString,
            offload__pb2.ServerOutput.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
