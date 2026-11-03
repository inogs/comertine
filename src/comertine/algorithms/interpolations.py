def linearly_interpolate_between(p1, p2, q1, q2, t):
    return (q1 * (p2 - t) + q2 * (t - p1)) / (p2 - p1)


def bilinear_interpolation(
        lat_min, lat_max, lon_min, lon_max, v_min_min, v_max_min,
        v_min_max, v_max_max, lat_position, lon_position):
    q1 = linearly_interpolate_between(lat_min, lat_max, v_min_min, v_max_min, lat_position)
    q2 = linearly_interpolate_between(lat_min, lat_max, v_min_max, v_max_max, lat_position)
    return linearly_interpolate_between(lon_min, lon_max, q1, q2, lon_position)
