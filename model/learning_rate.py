import numpy as np


def gen_lr_schedule(lr_init, decay_start, decay_end, steps_per_epoch):
    """
    Generates a linear learning rate schedule compatible with
    `tf.train.piecewise_constant`

    Keeps the same learning rate for the first <decay_start> epochs 
    and linearly decay the rate to zero until <decay_end> epochs. A
    rough ascii illustration:
    
    
             | ________________
             |                 \
             |                  \
          lr |                   \
             |                    \
             |--------------------------
                       epochs

    - lr_init: the initial learning rate
    - decay_start: number of epochs before learning rate reduction begins
    - decay_end: the epoch on which learning rate reaches zero
    - steps_per_epoch: the number of tf global_step increments in an epoch
    """
    def get_lr(current_epoch, final_epoch, lr_init):
        return lr_init - ((lr_init / final_epoch) * current_epoch)
    
    decay_interval = decay_end - decay_start + 1
    epochs = np.arange(1, decay_interval, dtype=np.int32)
    boundaries = list(steps_per_epoch * (epochs + decay_start))[:-1]
    values = [get_lr(e, decay_interval, lr_init) for e in epochs]
    return boundaries, values
