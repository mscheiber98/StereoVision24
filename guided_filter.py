import cv2
import numpy as np

## Adapted from: https://github.com/lisabug/guided-filter
## Implements the paper K.He, J.Sun, and X.Tang. Guided Image Filtering. TPAMI'12.

def to_32F(image):
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(np.float32(image), 0, 1)

def padding_constant(image, pad_size, constant_value=0):
    """
    Padding with constant value.

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height and width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))
    ret[h:-h, w:-w, :] = image

    ret[:h, :, :] = constant_value
    ret[-h:, :, :] = constant_value
    ret[:, :w, :] = constant_value
    ret[:, -w:, :] = constant_value
    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_reflect(image, pad_size):
    """
    Padding with reflection to image by boarder

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in range(shape[0]+2*h):
        for j in range(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[h-1-i, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h-1-i, j-w, :]
                else:
                    ret[i, j, :] = image[h-1-i, w+2*shape[1]-1-j, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, w+2*shape[1]-1-j, :]
            else:
                if j < w:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, j-w, :]
                else:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, w+2*shape[1]-1-j, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_reflect_101(image, pad_size):
    """
    Padding with reflection to image by boarder

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in range(shape[0]+2*h):
        for j in range(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[h-i, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h-i, j-w, :]
                else:
                    ret[i, j, :] = image[h-i, w+2*shape[1]-2-j, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, w+2*shape[1]-2-j, :]
            else:
                if j < w:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, j-w, :]
                else:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, w+2*shape[1]-2-j, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_edge(image, pad_size):
    """
    Padding with edge

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in range(shape[0]+2*h):
        for j in range(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[0, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[0, j-w, :]
                else:
                    ret[i, j, :] = image[0, shape[1]-1, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, shape[1]-1, :]
            else:
                if j < w:
                    ret[i, j, :] = image[shape[0]-1, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[shape[0]-1, j-w, :]
                else:
                    ret[i, j, :] = image[shape[0]-1, shape[1]-1, :]

    return ret if is_3D else np.squeeze(ret, axis=2)

def box_filter(I, r, normalize=True, border_type='reflect_101'):
    """

    Parameters
    ----------
    I: NDArray
        Input should be 3D with format of HWC
    r: int
        radius of filter. kernel size = 2 * r + 1
    normalize: bool
        Whether to normalize
    border_type: str
        Border type for padding, includes:
        edge        :   aaaaaa|abcdefg|gggggg
        zero        :   000000|abcdefg|000000
        reflect     :   fedcba|abcdefg|gfedcb
        reflect_101 :   gfedcb|abcdefg|fedcba

    Returns
    -------
    ret: NDArray
        Output has same shape with input
    """
    I = I.astype(np.float32)
    shape = I.shape
    assert len(shape) in [2, 3], \
        "I should be NDArray of 2D or 3D, not %dD" % len(shape)
    is_3D = True

    if len(shape) == 2:
        I = np.expand_dims(I, axis=2)
        shape = I.shape
        is_3D = False

    (rows, cols, channels) = shape

    tmp = np.zeros(shape=(rows, cols+2*r, channels), dtype=np.float32)
    ret = np.zeros(shape=shape, dtype=np.float32)

    # padding
    if border_type == 'reflect_101':
        I = padding_reflect_101(I, pad_size=(r, r))
    elif border_type == 'reflect':
        I = padding_reflect(I, pad_size=(r, r))
    elif border_type == 'edge':
        I = padding_edge(I, pad_size=(r, r))
    elif border_type == 'zero':
        I = padding_constant(I, pad_size=(r, r), constant_value=0)
    else:
        raise NotImplementedError

    I_cum = np.cumsum(I, axis=0) # (rows+2r, cols+2r)
    tmp[0, :, :] = I_cum[2*r, :, :]
    tmp[1:rows, :, :] = I_cum[2*r+1:2*r+rows, :, :] - I_cum[0:rows-1, :, :]

    I_cum = np.cumsum(tmp, axis=1)
    ret[:, 0, :] = I_cum[:, 2*r, :]
    ret[:, 1:cols, :] = I_cum[:, 2*r+1:2*r+cols, :] - I_cum[:, 0:cols-1, :]
    if normalize:
        ret /= float((2*r+1) ** 2)

    return ret if is_3D else np.squeeze(ret, axis=2)


class GuidedFilter:
    """
    This is a factory class which builds guided filter
    according to the channel number of guided Input.
    The guided input could be gray image, color image,
    or multi-dimensional feature map.

    References:
        K.He, J.Sun, and X.Tang. Guided Image Filtering. TPAMI'12.
    """
    def __init__(self, I, radius, eps):
        """

        Parameters
        ----------
        I: NDArray
            Guided image or guided feature map
        radius: int
            Radius of filter
        eps: float
            Value controlling sharpness
        """
        if len(I.shape) == 2:
            self._Filter = GrayGuidedFilter(I, radius, eps)
        else:
            self._Filter = MultiDimGuidedFilter(I, radius, eps)

    def filter(self, p):
        """

        Parameters
        ----------
        p: NDArray
            Filtering input which is 2D or 3D with format
            HW or HWC

        Returns
        -------
        ret: NDArray
            Filtering output whose shape is same with input
        """
        p = to_32F(p)
        if len(p.shape) == 2:
            return self._Filter.filter(p)
        elif len(p.shape) == 3:
            channels = p.shape[2]
            ret = np.zeros_like(p, dtype=np.float32)
            for c in range(channels):
                ret[:, :, c] = self._Filter.filter(p[:, :, c])
            return ret


class GrayGuidedFilter:
    """
    Specific guided filter for gray guided image.
    """
    def __init__(self, I, radius, eps):
        """

        Parameters
        ----------
        I: NDArray
            2D guided image
        radius: int
            Radius of filter
        eps: float
            Value controlling sharpness
        """
        self.I = to_32F(I)
        self.radius = radius
        self.eps = eps

    def filter(self, p):
        """

        Parameters
        ----------
        p: NDArray
            Filtering input of 2D

        Returns
        -------
        q: NDArray
            Filtering output of 2D

        p = cost_volume
        I = left image

        """
        # step 1
        meanI  = box_filter(I=self.I, r=self.radius)
        meanp  = box_filter(I=p, r=self.radius)
        corrI  = box_filter(I=self.I * self.I, r=self.radius)
        corrIp = box_filter(I=self.I * p, r=self.radius)
        # step 2
        varI   = corrI - meanI * meanI
        covIp  = corrIp - meanI * meanp
        # step 3
        a      = covIp / (varI + self.eps)
        b      = meanp - a * meanI
        # step 4
        meana  = box_filter(I=a, r=self.radius)
        meanb  = box_filter(I=b, r=self.radius)
        # step 5
        q = meana * self.I + meanb

        return q
    
class MultiDimGuidedFilter:
    """
    Specific guided filter for color guided image
    or multi-dimensional feature map.
    """
    def __init__(self, I, radius, eps):
        self.I = to_32F(I)
        self.radius = radius
        self.eps = eps

        self.rows = self.I.shape[0]
        self.cols = self.I.shape[1]
        self.chs  = self.I.shape[2]

    def filter(self, p):
        """

        Parameters
        ----------
        p: NDArray
            Filtering input of 2D

        Returns
        -------
        q: NDArray
            Filtering output of 2D
        """
        p_ = np.expand_dims(p, axis=2)

        meanI = box_filter(I=self.I, r=self.radius) # (H, W, C)
        meanp = box_filter(I=p_, r=self.radius) # (H, W, 1)
        I_ = self.I.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)
        meanI_ = meanI.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)

        corrI_ = np.matmul(I_, I_.transpose(0, 2, 1))  # (HW, C, C)
        corrI_ = corrI_.reshape((self.rows, self.cols, self.chs*self.chs)) # (H, W, CC)
        corrI_ = box_filter(I=corrI_, r=self.radius)
        corrI = corrI_.reshape((self.rows*self.cols, self.chs, self.chs)) # (HW, C, C)
        corrI = corrI - np.matmul(meanI_, meanI_.transpose(0, 2, 1))

        U = np.expand_dims(np.eye(self.chs, dtype=np.float32), axis=0)
        # U = np.tile(U, (self.rows*self.cols, 1, 1)) # (HW, C, C)

        left = np.linalg.inv(corrI + self.eps * U) # (HW, C, C)

        corrIp = box_filter(I=self.I*p_, r=self.radius) # (H, W, C)
        covIp = corrIp - meanI * meanp # (H, W, C)
        right = covIp.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)

        a = np.matmul(left, right) # (HW, C, 1)
        axmeanI = np.matmul(a.transpose((0, 2, 1)), meanI_) # (HW, 1, 1)
        axmeanI = axmeanI.reshape((self.rows, self.cols, 1))
        b = meanp - axmeanI # (H, W, 1)
        a = a.reshape((self.rows, self.cols, self.chs))

        meana = box_filter(I=a, r=self.radius)
        meanb = box_filter(I=b, r=self.radius)

        meana = meana.reshape((self.rows*self.cols, 1, self.chs))
        meanb = meanb.reshape((self.rows*self.cols, 1, 1))
        I_ = self.I.reshape((self.rows*self.cols, self.chs, 1))

        q = np.matmul(meana, I_) + meanb
        q = q.reshape((self.rows, self.cols))

        return q