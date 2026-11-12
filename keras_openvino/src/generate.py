import openvino as ov
import openvino.opset16 as ov_opset

from keras.src import tree
from keras_openvino.src.ops.core import convert_to_numpy
from keras_openvino.src.utils import get_device


def _input_signature(inputs):
    # The batch dimension is traced as dynamic, so it is left out.
    signature = []
    for x in tree.flatten(inputs):
        shape = getattr(x, "shape", None)
        if shape is not None:
            shape = tuple(shape)
            if shape:
                shape = (None,) + shape[1:]
        signature.append((shape, str(getattr(x, "dtype", None))))
    return tuple(signature)


def _compile(model, inputs, fn, device):
    struct_params = model._parameterize_data(inputs)
    struct_outputs = fn(struct_params)
    parameters = [p.output.get_node() for p in tree.flatten(struct_params)]
    results = [
        ov_opset.result(r.output)
        for r in tree.flatten(struct_outputs)
        if r is not None
    ]
    ov_model = ov.Model(results=results, parameters=parameters)
    ov_model.validate_nodes_and_infer_types()
    precision = "f16" if model.dtype in ("float16", "bfloat16") else "f32"
    compiled = ov.Core().compile_model(
        ov_model, device, {"INFERENCE_PRECISION_HINT": precision}
    )
    return compiled, struct_outputs


def _run(model, inputs, compiled, struct_outputs):
    raw = iter(compiled(tree.flatten(inputs)).to_tuple())
    flat_outputs = [
        None if x is None else next(raw) for x in tree.flatten(struct_outputs)
    ]
    packed = tree.pack_sequence_as(struct_outputs, flat_outputs)
    return model._unpack_singleton(packed)


def make_generate_function(model, generate_step, static_argnames=()):
    """Compile a KerasHub `generate_step` into an OpenVINO model.

    Arguments named in `static_argnames` are baked into the graph as
    constants, and any other keyword argument is traced as a model input.
    The compiled model is reused while the input shapes and the static
    arguments stay the same.
    """
    cache = {}

    def generate_function(inputs, **kwargs):
        static = {}
        for name in static_argnames:
            if name in kwargs:
                value = kwargs.pop(name)
                static[name] = (
                    tuple(value) if isinstance(value, list) else value
                )
        dynamic_names = tuple(sorted(kwargs))
        if dynamic_names:
            inputs = {**inputs, **kwargs}
        inputs = tree.map_structure(convert_to_numpy, inputs)

        def fn(x):
            dynamic = {}
            if dynamic_names:
                x = dict(x)
                dynamic = {name: x.pop(name) for name in dynamic_names}
            return generate_step(x, **static, **dynamic)

        device = get_device()
        signature = (
            device,
            _input_signature(inputs),
            tuple(sorted(static.items())),
            dynamic_names,
        )
        if cache.get("signature") != signature:
            cache["compiled"], cache["outputs"] = _compile(
                model, inputs, fn, device
            )
            cache["signature"] = signature
        return _run(model, inputs, cache["compiled"], cache["outputs"])

    return generate_function
