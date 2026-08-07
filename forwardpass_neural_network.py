# A single expression encoding the entire forward pass of a 3-layer network
# (Dense -> ReLU -> Dense -> ReLU -> Dense -> Softmax) plus the categorical
# cross-entropy loss, all nested into one np.dot/np.maximum/np.exp chain
# rather than split into separate layer objects. Relies on X, w1/b1, w2/b2,
# w3/b3, and y being defined elsewhere (this is a conceptual/illustrative
# snippet, not meant to run standalone).
import numpy as np


# T == Transpose

loss = -np.log(
    np.sum(
        y * np.exp(
            np.dot(np.maximum(
                0,
                np.dot(
                    np.maximum(
                        0,
                        np.dot(
                            X,
                            w1.T
                        ) +b1
                    ),
                ) + b2
            ),
                   w3.T
                   ) + b3
        ) /
        np.sum(
            np.exp(
                np.dot(
                    np.maximum(
                        0,
                        np.dot(
                            np.maximum(
                                0,
                                np.dot(
                                    X,
                                    w1.T
                                ) + b1
                            ),
                            w2.T
                        ) + b2
                    ),
                    w3.T
                ) + b3
            ),
            axis=1,
            keepdims=True
        )
    )
               )