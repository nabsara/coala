from typing import Literal, Optional

import numpy as np
import json
import pickle

try:
    import torch
except:
    pass

from .generated import DataExample as DataExamplePB

DTYPE_TO_PRECISION = {
    np.int16: DataExamplePB.Precision.INT16,
    np.int32: DataExamplePB.Precision.INT32,
    np.int64: DataExamplePB.Precision.INT64,
    np.float16: DataExamplePB.Precision.FLOAT16,
    np.float32: DataExamplePB.Precision.FLOAT32,
    np.float64: DataExamplePB.Precision.FLOAT64,
}

PRECISION_TO_DTYPE = {
    DataExamplePB.Precision.INT16: np.int16,
    DataExamplePB.Precision.INT32: np.int32,
    DataExamplePB.Precision.INT64: np.int64,
    DataExamplePB.Precision.FLOAT16: np.float16,
    DataExamplePB.Precision.FLOAT32: np.float32,
    DataExamplePB.Precision.FLOAT64: np.float64,
}

def dict_to_buffer(metadata):
    for k, v in metadata.items():
        if isinstance(v, set):
            metadata[k] = list(v)

    data = json.dumps(metadata).encode('utf-8')
    return data


def dict_from_buffer(b):
    metadata = json.loads(b.data.decode())
    return metadata


class DataExample(object):

    def __init__(
            self,
            b: Optional[str] = None,
            output_type: Literal["numpy", "torch"] = "numpy") -> None:
        if b is not None:
            self.ex = DataExamplePB.FromString(b)
        else:
            self.ex = DataExamplePB()

        self.output_type = output_type

    def get(self, key: str):
        if key == "midi":
            midi = self.ex.buffers["midi"].data
            midi = pickle.loads(midi)
            return midi
        
        buf = self.ex.buffers[key]
        if buf is None:
            raise KeyError(f"key '{key}' not available")

        array = np.frombuffer(
            buf.data,
            dtype=PRECISION_TO_DTYPE[buf.precision],
        ).reshape(buf.shape).copy()

        if self.output_type == "numpy":
            pass
        elif self.output_type == "torch":
            array = torch.from_numpy(array)
        else:
            raise ValueError(f"Output type {self.output_type} not available")

        return array

    def put(self, key: str, array: np.ndarray, dtype: np.dtype):
        buffer = self.ex.buffers[key]

        buffer.data = np.asarray(array).astype(dtype).tobytes()
        buffer.shape.extend(array.shape)
        buffer.precision = DTYPE_TO_PRECISION[dtype]

    def put_metadata(self, metadata: dict):
        meta_buffer = self.ex.buffers["metadata"]
        meta_buffer.data = dict_to_buffer(metadata)

    def get_metadata(self):
        buf = self.ex.buffers["metadata"]
        return dict_from_buffer(buf)

    def put_buffer(self, key: str, b: bytes, shape: list):
        buffer = self.ex.buffers[key]
        buffer.data = b
        if shape is not None:
            buffer.shape.extend(shape)
        buffer.precision = DataExamplePB.Precision.INT16

    def put_array(self, key: str, array: np.ndarray, dtype: np.dtype):
        buffer = self.ex.buffers[key]
        buffer.data = np.asarray(array).astype(dtype).tobytes()
        for i in range(len(buffer.shape)):
            buffer.shape.pop()
        buffer.shape.extend(array.shape)
        buffer.precision = DTYPE_TO_PRECISION[dtype]

    def as_dict(self):
        return {k: self.get(k) for k in self.ex.buffers}

    def get_keys(self):
        return list(self.ex.buffers.keys())
    
    def __str__(self) -> str:
        repr = []
        repr.append("DataExample(")
        for key in self.ex.buffers:
            if key == "metadata":
                repr.append(str(self.get_metadata()))
            else:
                array = self.get(key)
                repr.append(f"\t{key}[{array.dtype}] {array.shape},")
        repr.append(")")
        return "\n".join(repr)

    def __bytes__(self) -> str:
        return self.ex.SerializeToString()
