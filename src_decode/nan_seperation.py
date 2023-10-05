from numba import njit
from numba.typed import List
import numpy as np


@njit
def seperate_by_nans(a, default_size):
    arr = np.zeros(default_size)
    ls = List()
    j = 0

    is_nan = np.isnan(a)
    for i in range(len(is_nan)):
        if is_nan[i]:
            # finish current array, start next array
            ls.append(arr[arr != 0])
            arr = np.zeros(default_size)
            j = 0
        else:
            if j >= default_size:
                print("[nan_seperation] error, too many counts. enlarge default size")
                return ls
            arr[j] = a[i]
            j = j + 1

    return ls

@njit
def seperate_by_nans_2d(a, default_size):
    # assuming a is n x 2
    arr = np.zeros((default_size, 2))
    ls = List()
    j = 0

    is_nan = np.isnan(a[:,0])
    for i in range(len(is_nan)):
        if is_nan[i]:
            # finish current array, start next array
            ls.append(arr[arr[:,0] != 0,:])
            # ls.append(arr)
            arr = np.zeros((default_size, 2))
            j = 0
        else:
            if j >= default_size:
                print("[nan_seperation] error, too many counts. enlarge default size")
                return ls
            arr[j] = a[i]
            j = j + 1

    return ls


if __name__ == "__main__":
    #nan = np.nan
    a = np.array([3, 4, np.nan, 7, 4, np.nan, np.nan, 5, 23, 1, 4, 5, 43, np.nan, np.nan, np.nan, 5, np.nan, 5, 6, 4])
    print(a)
    result = seperate_by_nans(a, 15)

    print(result[0])
    print(result)
    print(type(result[0]))
