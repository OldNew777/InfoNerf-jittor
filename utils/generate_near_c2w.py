import numpy as np
import jittor as jt



def get_near_pixel(coords, padding):
    '''
    padding is the distance range (manhattan distance)

    '''
    N = coords.size(0)
    m_distance = np.random.randint(1, padding + 1)  # manhattan distance
    # get diff 
    x_distance = jt.randint(0, m_distance + 1, (N, 1))
    y_distance = m_distance - x_distance
    sign_ = jt.randint(0, 2, (N, 2)) * 2 - 1
    delta = jt.concat((x_distance, y_distance), dim=1)
    delta *= sign_
    # get near coords
    near_coords = coords + delta
    return near_coords
