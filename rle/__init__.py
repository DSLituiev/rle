import numpy as np
import numba

@numba.jit(nopython=True)
def _encode(stream):
    out = [0];
    oidx = 0;
    mcu = 0;
    midx = 0;
    symbol = False;
    for i in range(len(stream)+1):
        if (i < len(stream) and (not not stream[i]) == symbol):
            mcu+=1
        else:
            for s in range(4):
                k = (mcu >> (s * 7)) & 0x7f;
                out[oidx] |= k << (midx * 7 + 4);
                midx += 1
                if (midx == 4):
                    oidx+=1
                    out.append(0)
                    
                    midx = 0;
                # overflow?
                if (not (mcu > (1 << 7 * (s + 1)) - 1)):
                    break
                else:
                    out[oidx] |= 1 << midx;
            mcu = 1;
            symbol = not symbol
    return out

@numba.jit(nopython=True)
def _decode(stream):
    out = [];
    mcu = 0;
    symbol = True;
    midx = 0;
    for i in range(len(stream)):
        for s in range(4):
            overflow = stream[i] & (1 << s);
            if (not overflow):
                mcu -= 1
                while (mcu >= 0):
                    mcu -= 1
                    out.append(symbol)
                mcu = 0;
                symbol = ~symbol;
                midx = 0;
            else:
                midx+=1;
                if (midx == 4):
                    raise ValueError('Corrupted data.')
            k = (stream[i] >> (s * 7 + 4)) & 0x7f;
            mcu |= k << (7 * midx);
    mcu -= 1
    while (mcu >= 0):
        mcu -= 1
        out.append(symbol)
    return out

def decode(buffer_, shape=None):
    """encodes a byte sequence into a boolean array using RLE.
    Arguments:
    - buffer           -- a binary buffer
    - shape [optional] -- output shape
    """
    encoded_array = np.frombuffer(buffer_, dtype=np.uint32)
    decoded = np.asarray(_decode(encoded_array), dtype=bool)
    if shape is None:
        return decoded
    else:
        return decoded.reshape(shape)

def encode(x):
    "encodes a binary array into a byte sequence using RLE"
    x = np.asarray(x).ravel()
    encoded_array = np.asarray(_encode(x), dtype=np.uint32)
    return encoded_array.tobytes()
