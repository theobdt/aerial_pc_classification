from numpy import genfromtxt, savetxt


def read_pts(filename, delimiter=' '):
    data = genfromtxt(filename, delimiter=delimiter)
    return data


def write_pts(data, filename, delimiter=' '):
    savetxt(filename, data, fmt='%.2f')
