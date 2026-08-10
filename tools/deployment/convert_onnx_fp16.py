"""
Convert an exported DEIMv2 ONNX model to float16.

Besides the standard ``onnxconverter-common`` conversion, this fixes Cast nodes that the converter
leaves inconsistent: Casts whose input is not a float tensor (e.g. int64 shape values used for the
attention's 1/sqrt(d) scale) keep ``to=FLOAT`` while their declared output type becomes FLOAT16,
which makes the model fail to load. Those Casts are re-targeted to FLOAT16.

Usage::

    python tools/deployment/convert_onnx_fp16.py -i outputs/deimv2_dinov2_product/best_stg2.onnx
"""

import argparse

import onnx
from onnx import TensorProto
from onnxconverter_common import float16


def convert(input_path: str, output_path: str) -> None:
    """Convert ``input_path`` to a float16 ONNX model saved at ``output_path``.

    :param input_path: Path to the float32 ONNX model.
    :param output_path: Path where the float16 model is written.
    """
    model = onnx.load(input_path)
    # Refresh shape/type annotations; stale value_info confuses the converter.
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)

    # keep_io_types: the model still takes float32 images / int64 sizes and returns float32 outputs.
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

    value_types = {
        v.name: v.type.tensor_type.elem_type
        for v in list(model_fp16.graph.value_info) + list(model_fp16.graph.output)
    }
    patched = 0
    for node in model_fp16.graph.node:
        if node.op_type != 'Cast':
            continue
        to_attr = next(attr for attr in node.attribute if attr.name == 'to')
        if to_attr.i == TensorProto.FLOAT and value_types.get(node.output[0]) == TensorProto.FLOAT16:
            to_attr.i = TensorProto.FLOAT16
            patched += 1
    print(f'patched {patched} Cast nodes')

    onnx.save(model_fp16, output_path)
    onnx.checker.check_model(output_path)
    print(f'saved {output_path} (checker passed)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', '-i', type=str, required=True, help='float32 ONNX model')
    parser.add_argument('--output', '-o', type=str, help="output path (default: replace '.onnx' with '_fp16.onnx')")
    args = parser.parse_args()
    convert(args.input, args.output or args.input.replace('.onnx', '_fp16.onnx'))
