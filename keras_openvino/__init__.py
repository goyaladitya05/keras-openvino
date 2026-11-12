from keras_openvino.src.version import __version__


def make_generate_function(model, generate_step, static_argnames=()):
    # Imported lazily, as `keras_openvino` can be imported before Keras.
    from keras_openvino.src.generate import make_generate_function

    return make_generate_function(model, generate_step, static_argnames)
