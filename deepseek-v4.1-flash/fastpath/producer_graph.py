"""quant_producer: boot-only Driver introspection of the captured CUDA graphs (no device work after startup)."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json


@contextmanager
def retained_graphs():
    import torch
    original = torch.cuda.CUDAGraph

    class RetainedGraph(original):
        def __new__(cls, keep_graph=False):
            return super().__new__(cls, True)

        def __init__(self, keep_graph=False):
            super().__init__(keep_graph=True)

        def capture_end(self):
            super().capture_end()
            self.instantiate()  # instantiate at capture end, as an unretained graph does; no lazy first replay

    torch.cuda.CUDAGraph = RetainedGraph
    try:
        yield
    finally:
        torch.cuda.CUDAGraph = original


def _ok(result):
    from cuda.bindings import driver as d
    if result[0] != d.CUresult.CUDA_SUCCESS:
        raise RuntimeError('Driver graph inspection failed: ' + repr(result[0]))
    values = result[1:]
    return values[0] if len(values) == 1 else values


def _json(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'))


def inventory(graph):
    from cuda.bindings import driver as d
    functions = {}
    quant_names = set()

    def function(func):
        key = int(func)
        if key not in functions:
            name = _ok(d.cuFuncGetName(func))
            if isinstance(name, bytes): name = name.decode()
            attrs = [int(_ok(d.cuFuncGetAttribute(a, func))) for a in (
                d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)]
            functions[key] = (name, attrs)
        return functions[key]

    def walk(handle):
        handle = d.CUgraph(init_value=int(handle))
        _, count = _ok(d.cuGraphGetNodes(handle, numNodes=0))
        nodes, got = _ok(d.cuGraphGetNodes(handle, numNodes=count))
        if got != count: raise RuntimeError('graph changed during boot inspection')
        labels = {}
        node_counts = Counter()
        children = []
        for node in nodes:
            kind = _ok(d.cuGraphNodeGetType(node))
            label = [int(kind)]
            if kind == d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                p = _ok(d.cuGraphKernelNodeGetParams(node))
                name, attrs = function(p.func)
                grid = [p.gridDimX, p.gridDimY, p.gridDimZ]
                block = [p.blockDimX, p.blockDimY, p.blockDimZ]
                # The private kernel adds only an Int32 flag and notification.
                # All launch geometry/resources stay in the comparison key.
                selected = ('MXFP8QuantizeSwizzledKernel' in name and '5120' in name
                            and 'tensorptrbf16gmem' in name and block == [640, 1, 1]
                            and int(p.sharedMemBytes) == 0)
                if selected:
                    quant_names.add(name)
                    name = '__BF16_K5120_SWIZZLED_4T_QUANT__'
                if ('seam_kernelILi16' in name and block == [256,1,1]
                        and int(p.sharedMemBytes) == 0 and attrs == [122,29452,0]):
                    name = '__QUALIFIED_CS16_INPUT_QUANT_SEAM__'
                label += [name, grid, block, int(p.sharedMemBytes), attrs]
            elif kind == d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH:
                child = walk(_ok(d.cuGraphChildGraphNodeGetGraph(node)))
                children.append(child)
                label += [child['sha256']]
            label = _json(label)
            labels[int(node)] = label
            node_counts[label] += 1
        # Query typed edges; silently dropping PDL port/type information is forbidden.
        _, _, _, edge_count = _ok(d.cuGraphGetEdges(handle, numEdges=0))
        edges = Counter()
        if edge_count:
            src, dst, data, got = _ok(d.cuGraphGetEdges(handle, numEdges=edge_count))
            if got != edge_count: raise RuntimeError('graph edges changed during inspection')
            for a, b, e in zip(src, dst, data):
                edges[_json([labels[int(a)], labels[int(b)], int(e.from_port),
                             int(e.to_port), int(e.type)])] += 1
        canonical = dict(nodes=sorted(node_counts.items()), edges=sorted(edges.items()))
        return dict(node_count=int(count), edge_count=int(edge_count),
                    node_inventory=dict(node_counts), edge_inventory=dict(edges),
                    child_graphs=children,
                    sha256=hashlib.sha256(_json(canonical).encode()).hexdigest())

    result = walk(graph.raw_cuda_graph())
    result['raw_quant_names'] = sorted(quant_names)
    return result


def is_selected_input_quant(name, block):
    return ('MXFP8QuantizeSwizzledKernel' in name and '5120' in name
            and 'tensorptrbf16gmem' in name and list(block)==[640,1,1])


def last_kernel():
    import torch
    from cuda.bindings import driver as d
    status, capture_id, graph, dependencies, edge_data, count = _ok(d.cuStreamGetCaptureInfo(
        d.CUstream(init_value=int(torch.cuda.current_stream().cuda_stream))))
    if status != d.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE or count != 1:
        raise RuntimeError('producer quant capture has no unique last dependency')
    node = dependencies[0]
    if _ok(d.cuGraphNodeGetType(node)) != d.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
        raise RuntimeError('producer quant last dependency is not a kernel')
    params = _ok(d.cuGraphKernelNodeGetParams(node))
    name = _ok(d.cuFuncGetName(params.func))
    if isinstance(name, bytes): name = name.decode()
    if not is_selected_input_quant(name, (params.blockDimX,params.blockDimY,params.blockDimZ)):
        raise RuntimeError('selected quant call did not capture its expected kernel')
    return int(graph), int(node)
