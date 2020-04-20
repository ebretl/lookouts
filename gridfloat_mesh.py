import time
from xml.etree import ElementTree
import struct
import sys
import os
import pickle
from collections import deque
import uuid
import glob

import pygeodesy
import numpy as np
import matplotlib.pyplot as plt


def get_or_cache(name, func, *args, **kwargs):
    osp = os.path
    cache_id = uuid.uuid3(uuid.NAMESPACE_DNS, name)
    store_path = osp.join(osp.dirname(__file__), "data_stores",
                          str(cache_id) + ".pkl")
    if osp.exists(store_path):
        with open(store_path, 'rb') as file:
            return pickle.load(file)
    os.makedirs(osp.dirname(store_path), exist_ok=True)
    result = func(*args, **kwargs)
    with open(store_path, 'wb') as file:
        pickle.dump(result, file)
    return result


def make_grid_geo(data_dir):
    meta_file_path = glob.glob(os.path.join(data_dir, "*meta.xml"))
    assert len(meta_file_path) == 1
    meta_file_path = meta_file_path[0]

    floatgrid_file_path = glob.glob(os.path.join(data_dir, "*.flt"))
    assert len(floatgrid_file_path) == 1
    floatgrid_file_path = floatgrid_file_path[0]

    header_file_path = glob.glob(os.path.join(data_dir, "*.hdr"))
    assert len(header_file_path) == 1
    header_file_path = header_file_path[0]

    xml_tree = ElementTree.parse(meta_file_path)
    bounds = dict()
    for bound_xml in xml_tree.findall("idinfo/spdom/bounding/"):
        bounds[bound_xml.tag] = float(bound_xml.text)

    n_rows = int(xml_tree.find("spdoinfo/rastinfo/rowcount").text)
    n_cols = int(xml_tree.find("spdoinfo/rastinfo/colcount").text)

    with open(header_file_path) as f:
        for line in f:
            if line.startswith("byteorder"):
                order = line.replace("byteorder", "").strip().lower()
                if order == 'lsbfirst':
                    unpack_fmt = '<' + str(n_rows * n_cols) + 'f'
                elif order == 'msbfirst':
                    unpack_fmt = '>' + str(n_rows * n_cols) + 'f'
                else:
                    print("ERROR: unexpected byteorder")
                    sys.exit(1)

    with open(floatgrid_file_path, 'rb') as f:
        flat_data = struct.unpack(unpack_fmt, f.read())

    lats = np.linspace(bounds["northbc"], bounds["southbc"], n_rows,
                       endpoint=False, dtype=np.float64)
    lons = np.linspace(bounds["westbc"], bounds["eastbc"], n_cols,
                       endpoint=False, dtype=np.float64)
    grid_geodetic = np.zeros((n_rows, n_cols, 3),
                             dtype=np.float64)  # lat, lon, elevation
    grid_geodetic[:, :, 0] = np.column_stack([lats] * n_cols)
    grid_geodetic[:, :, 1] = np.stack([lons] * n_cols)
    grid_geodetic[:, :, 2] = np.array(flat_data).reshape((n_rows, n_cols))
    return grid_geodetic


def point_in_triangle(p, tri):
    assert tri.shape == (3, 3)
    plane_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    plane_normal /= np.dot(plane_normal, plane_normal) ** 0.5
    assert (p - tri[0]).dot(plane_normal) < 1e-6
    return all(np.cross(plane_normal, tri[i] - tri[i - 1]).dot(p - tri[i]) >= 0
               for i in range(3))


def ray_hits_triangle(origin, r, tri):
    assert tri.shape == (3, 3)
    # print(tri.round(2))
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    n /= np.dot(n, n) ** 0.5
    r_dot_n = np.dot(r, n)
    if abs(r_dot_n) < 1e-7:
        print("r dot n too small", r_dot_n)
        return False
    t = np.dot(tri[0] - origin, n) / r_dot_n
    # print("testing point", origin + r * t)
    # print("t", t)
    return t >= 0 and point_in_triangle(origin + r * t, tri)


def ray_hits_quad(origin, r, quad):
    return (ray_hits_triangle(origin, r, quad[:3])
            or ray_hits_triangle(origin, r, quad[[2, 3, 0]]))


def ray_passes_over_triangle(origin, r, tri):
    assert tri.shape[0] == 3
    r_perp = np.array([-r[1], r[0]])
    v = tri[:, :2] - origin[:2]
    d = np.dot(v, r[:2])
    s = np.dot(v, r_perp)
    return np.any(d > 0) and (np.count_nonzero(s > 0) not in [0, 3])


def ray_passes_over_quad(origin, r, quad):
    return (ray_passes_over_triangle(origin, r, quad[:3])
            or ray_passes_over_triangle(origin, r, quad[[2, 3, 0]]))


if __name__ == '__main__':
    data_dir = os.path.expanduser("~/Downloads/n41w080")
    grid_geo = get_or_cache(data_dir, make_grid_geo, data_dir)

    print(grid_geo[0, 0], grid_geo[-1, -1])

    def make_grid_ecef():
        geo_ecef_converter = pygeodesy.ecef.EcefKarney(pygeodesy.Datums.WGS84)
        grid_ecef = np.zeros_like(grid_geo)  # x, y, z
        for r in range(grid_geo.shape[0]):
            for c in range(grid_geo.shape[1]):
                grid_ecef[r, c] = geo_ecef_converter.forward(*grid_geo[r, c])[
                                  :3]
        return grid_ecef


    grid_ecef = get_or_cache("ecef_n41w080", make_grid_ecef)

    ref_r = 2378
    ref_c = 2782
    ref_pt = grid_ecef[ref_r, ref_c]
    local_up = ref_pt / np.linalg.norm(ref_pt)
    local_north = np.array((0, 0, 1), dtype=local_up.dtype)
    local_north -= (local_up * local_north.dot(local_up))
    local_north /= np.linalg.norm(local_north)
    local_east = np.cross(local_north, local_up)
    print("east", local_east, "north", local_north, "up", local_up)
    R = np.column_stack([local_east, local_north, local_up])
    local_coords = (grid_ecef - ref_pt).dot(R)


    def quad_at(r, c):
        return local_coords[[r, r + 1, r + 1, r], [c, c, c + 1, c + 1]]


    view_area = np.zeros(local_coords.shape[:2], dtype=np.bool)
    visited = view_area.copy()


    def view_dist(start_r, start_c, yaw, grade, height):
        origin = local_coords[start_r, start_c] + np.array([0, 0, height])
        ray = np.array([np.cos(yaw), np.sin(yaw), grade])

        bfs = deque([(start_r, start_c)])
        visited[:, :] = False
        visited[start_r, start_c] = True
        done = False
        result = -1

        quad = quad_at(start_r, start_c)
        if ray_hits_quad(origin, ray, quad):
            result = np.linalg.norm(quad.mean(axis=0) - origin)
            done = True

        while len(bfs) > 0 and not done:
            r1, c1 = bfs.popleft()
            for r2, c2 in [(r1 + 1, c1), (r1, c1 + 1), (r1 - 1, c1),
                           (r1, c1 - 1)]:
                if (r2 < 0 or r2 + 1 >= visited.shape[0] or c2 < 0
                        or c2 + 1 >= visited.shape[1] or visited[r2, c2]):
                    continue

                quad = quad_at(r2, c2)
                if ray_passes_over_quad(origin, ray, quad):
                    if ray_hits_quad(origin, ray, quad):
                        result = np.linalg.norm(quad.mean(axis=0) - origin)
                        done = True
                    bfs.append((r2, c2))
                visited[r2, c2] = True
        # print("time", time.time() - t0)
        # print(result)
        # plt.imshow(visited)
        # plt.show()
        view_area[visited] = True
        return result


    t0 = time.time()
    yaws = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    view_dists = [view_dist(ref_r, ref_c, yaw, -0.01, 1) for yaw in yaws]
    print(np.mean(view_dists))
    print("time", time.time() - t0)

    plt.subplot(1, 2, 1)
    plt.imshow(grid_geo[:, :, 2])
    plt.subplot(1, 2, 2)
    plt.imshow(view_area)
    plt.show()
