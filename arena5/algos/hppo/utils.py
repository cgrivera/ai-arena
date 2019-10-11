import numpy as np
import tensorflow as tf


def ned_to_ripCoords_tf(xyz, max_det):
    xyz = tf.cast(xyz, tf.float32)
    max_det = tf.cast(max_det, tf.float32)
    xy = tf.reduce_sum(tf.square(xyz[:, :, :, :1]), axis=-1)
    r_tmp = xy + tf.square(xyz[:, :, :, 2])
    r_tmp = (max_det - tf.clip_by_value(tf.sqrt(r_tmp), 0, max_det)) / max_det
    thet_tmp = tf.atan2(tf.sqrt(xy), xyz[:, :, :, 2])
    phi_tmp = tf.atan2(xyz[:, :, :, 1], xyz[:, :, :, 0])
    rip_crd = tf.stack([tf.sin(thet_tmp), tf.cos(thet_tmp), tf.sin(phi_tmp), tf.cos(phi_tmp)], axis=-1)
    rip_crd = tf.expand_dims(r_tmp, axis=-1) * rip_crd

    return rip_crd
