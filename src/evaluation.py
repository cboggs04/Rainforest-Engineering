import numpy as np
import tensorflow as tf


class TFLiteModel:
    def __init__(self, path, reference=False):
        kw = {}
        if reference:  # plain reference kernels (closest to TFLite Micro); no XNNPACK delegate
            kw['experimental_op_resolver_type'] = tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        self.it = tf.lite.Interpreter(model_path=path, **kw)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.scale, self.zp = self.inp["quantization"]
        self.oscale, self.ozp = self.out["quantization"]

    def predict(self, X):
        res = []
        for x in X:
            q = np.clip(np.round(x / self.scale + self.zp), -128, 127).astype(np.int8)[None]
            self.it.set_tensor(self.inp["index"], q)
            self.it.invoke()
            o = self.it.get_tensor(self.out["index"])[0].astype(np.float32)
            res.append((o - self.ozp) * self.oscale)
        return np.array(res)
