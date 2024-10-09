# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

rec_header_dtd = [
        ("nx", "i4"), ("ny", "i4"), ("nz", "i4"),
        ("mode", "i4"), ("nxstart", "i4"), ("nystart", "i4"), ("nzstart", "i4"),
        ("mx", "i4"), ("my", "i4"), ("mz", "i4"),
        ("xlen", "f4"), ("ylen", "f4"), ("zlen", "f4"),
        ("alpha", "f4"), ("beta", "f4"), ("gamma", "f4"),
        ("mapc", "i4"), ("mapr", "i4"), ("maps", "i4"),
        ("amin", "f4"), ("amax", "f4"), ("amean", "f4"),
        ("ispg", "i4"), ("next", "i4"), ("creatid", "i2"),
        ("extra_data", "V30"), ("nint", "i2"), ("nreal", "i2"),
        ("extra_data2", "V20"), ("imodStamp", "i4"), ("imodFlags", "i4"),
        ("idtype", "i2"), ("lens", "i2"), ("nphase", "i4"),
        ("vd1", "i2"), ("vd2", "i2"),
        ("triangles", "f4", 6), ("xorg", "f4"), ("yorg", "f4"), ("zorg", "f4"),
        ("cmap", "S4"), ("stamp", "u1", 4), ("rms", "f4"),
        ("nlabl", "i4"), ("labels", "S80", 10)
    ]

# Function to read MRC files
def read_mrc(filename, filetype='image'):    

    fd = open(filename, 'rb')
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)
    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]

    if header[0][3] == 1:
        data_type = 'int16'
    elif header[0][3] == 2:
        data_type = 'float32'
    elif header[0][3] == 4:
        data_type = 'float32'
        nx *= 2
    elif header[0][3] == 6:
        data_type = 'uint16'

    data = np.ndarray(shape=(nx, ny, nz), dtype=data_type)
    imgrawdata = np.fromfile(fd, data_type)
    fd.close()

    if filetype == 'image':
        for iz in range(nz):
            data_2d = imgrawdata[nx * ny * iz:nx * ny * (iz + 1)]
            data[:, :, iz] = data_2d.reshape(nx, ny, order='F')
    else:
        data = imgrawdata

    return header, data


def write_mrc(filename, img_data, header):

    if img_data.dtype == 'int16':
        header[0][3] = 1
    elif img_data.dtype == 'float32':
        header[0][3] = 2
    elif img_data.dtype == 'uint16':
        header[0][3] = 6

    fd = open(filename, 'wb')
    for i in range(len(rec_header_dtd)):
        header[rec_header_dtd[i][0]].tofile(fd)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]
    imgrawdata = np.ndarray(shape=(nx * ny * nz), dtype='uint16')
    for iz in range(nz):
        imgrawdata[nx * ny * iz:nx * ny * (iz + 1)] = img_data[:, :, iz].reshape(nx * ny, order='F')
    imgrawdata.tofile(fd)

    fd.close()
    return
