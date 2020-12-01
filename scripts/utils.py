import numpy as np
import tensorflow as tf


def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def point_cloud_2_birdseye(points, res=0.1, side_range=(-2.0, 2.0), fwd_range=(-2.0, 2.0), height_range=(0, 2.5)):
    """ Creates an 2D birds eye view representation of the point cloud data.
    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im


def generate_self_vector(pc_with_object_id):
    if len(pc_with_object_id.shape) == 3:
        pc_with_object_id = pc_with_object_id[:, :, -1]
    else:
        pc_with_object_id = pc_with_object_id[:, -1]

    temp = np.logical_not(np.logical_and(pc_with_object_id, np.ones(pc_with_object_id.shape)))
    pc_with_object_id = np.array(temp, dtype='int')
    return pc_with_object_id


def rmse_loss(targets, outputs):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(outputs, targets))))


def load_data(pc_filename, human_pos_filename, q_filename):

    x_train = np.load(pc_filename)
    y_train = np.load(human_pos_filename)
    q = np.load(q_filename)
    x_train_ = []
    for point_cloud in range(len(x_train)):
        for point in range(24):
            if x_train[point_cloud][point][-1] == 0:
                x_train[point_cloud][point][:-1] = np.zeros((3,))
    # x_train = np.reshape(x_train[:, :, :-1], (-1, 3))
    # print(x_train.shape)
    #  ZERO POINT CLOUD REMOVAL
    x_filtered = []
    y_filtered = []
    q_filtered = []
    for idx, each in enumerate(x_train[:, :, :-1]):
        if np.sum(each) != 0:
            x_filtered.append(each)
            y_filtered.append(y_train[idx].flatten())
            q_filtered.append(q[idx])
    x_filtered = np.vstack(x_filtered)
    #print(x_filtered.shape)
    x_filtered = np.reshape(x_filtered, (-1, 24, 3))
    x_filtered = np.reshape(x_filtered, (-1, 72))
    y_filtered = np.vstack(y_filtered)
    q_filtered = np.vstack(q_filtered)

    #plt.scatter(y_filtered[:, 0], y_filtered[:, 1])
    #plt.scatter(x_filtered[:, 0], x_filtered[:, 1])

    #plt.show()

    #x_train = np.hstack([x_filtered, q_filtered])
    x_train = x_filtered
    x_train_ = x_train
    x_train = np.hstack([x_filtered, q_filtered])

    #x_train = np.reshape(x_train, (-1, 78))
    y_train = y_filtered[:, :-1]

    # x_shuff = np.hstack([x_train, y_train])
    # np.random.shuffle(x_shuff)
    #
    # x_train = np.vstack([x_train, x_shuff[:, :-2]])
    # y_train = np.vstack([y_train, x_shuff[:, -2:]])
    #
    # x_shuff = np.hstack([x_train, y_train])
    # np.random.shuffle(x_shuff)
    #
    # x_train = np.vstack([x_train, x_shuff[:, :-2]])
    # y_train = np.vstack([y_train, x_shuff[:, -2:]])
    #
    # x_train__ = np.reshape(x_train[:, :-6], (-1,24,3))
    # x_train__ = x_train__ + np.random.randn(x_train__.shape[0], 24, 3)*0.1
    # x_train[:, :-6] = np.reshape(x_train__, (-1, 72))

    return x_train, y_train, x_train_

