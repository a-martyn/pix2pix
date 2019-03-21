import tensorflow as tf
tf.enable_eager_execution()


# Losses
# ----------------------------------

def bce_loss(y_true, y_pred):
    """ 
    Binary cross entropy loss.
    - y_true: array of floats between 0 and 1
    - y_preds: sigmoid activations output from model
    """
    EPS = 1e-12
    x = -tf.reduce_mean((y_true * tf.log(y_pred + EPS)) + ((1-y_true) * tf.log(1-y_pred + EPS)))
    return x


def l1_loss(y_true, y_pred):
    """ L1 Loss with mean reduction per PyTorch default """
    # abs(targets - outputs) => 0
    return tf.reduce_mean(tf.abs(y_true - y_pred))


# API
# ----------------------------------

def d_loss_fn(model, x, y_true):
    """
    Discriminators' loss function. Returns tensor for backprop. 
    """
    EPS = 1e-12
    y_pred = model([x[0], x[1]])
    loss_disc = bce_loss(y_true, y_pred)
    return loss_disc


def g_loss_fn(model, x, y_true, lambda_L1):
    """
    Generators' loss function. Returns tensor for backprop.   
    """
    g_pred, d_pred = model(x)
    loss_L1 = l1_loss(y_true[0], g_pred) * lambda_L1
    loss_gan = bce_loss(y_true[1], d_pred)
    loss_total = loss_gan + loss_L1
    return loss_total, loss_L1, loss_gan