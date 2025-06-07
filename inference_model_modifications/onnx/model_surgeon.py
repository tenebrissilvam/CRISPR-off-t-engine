import numpy as np
import onnx
import onnx_graphsurgeon as gs


def main():
    graph = gs.import_onnx(onnx.load("crispr_detector.onnx"))

    for tensor in graph.tensors().values():
        if isinstance(tensor, gs.Constant) and tensor.values.dtype == np.int64:
            tensor.values = tensor.values.astype(np.int32)

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "crispr_detector_fixed.onnx")

    print("Fixed model saved as crispr_detector_fixed.onnx")


if __name__ == "__main__":
    main()
