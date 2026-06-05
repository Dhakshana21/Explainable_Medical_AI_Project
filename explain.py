import tensorflow as tf
import numpy as np

def get_gradcam(model, img_array, layer_name="conv2d_2"):

    # Run the model once to ensure graph is built
    preds = model(img_array)

    # Get the convolution layer
    conv_layer = model.get_layer(layer_name)

    # Build GradCAM model safely
    grad_model = tf.keras.Model(
        inputs=model.layers[0].input,
        outputs=[conv_layer.output, model.layers[-1].output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:,0]

    grads = tape.gradient(loss, conv_outputs)

    # safety check
    if grads is None:
        return np.zeros((7,7))

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0)

    if tf.reduce_max(heatmap) != 0:
        heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()