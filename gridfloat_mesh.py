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

    lats = np.linspace(bounds["northbc"], bounds["southbc"], n_rows, endpoint=False,
                       dtype=np.float64)
    lons = np.linspace(bounds["westbc"], bounds["eastbc"], n_cols, endpoint=False, dtype=np.float64)
    grid_geodetic = np.zeros((n_rows, n_cols, 3), dtype=np.float64)  # lat, lon, elevation
    grid_geodetic[:, :, 0] = np.column_stack([lats] * n_cols)
    grid_geodetic[:, :, 1] = np.stack([lons] * n_cols)
    grid_geodetic[:, :, 2] = np.array(flat_data).reshape((n_rows, n_cols))
    return grid_geodetic


def point_in_triangle(p, tri):
    assert tri.shape == (3, 3)
    plane_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    plane_normal /= np.dot(plane_normal, plane_normal) ** 0.5
    assert (p - tri[0]).dot(plane_normal) < 1e-6
    return all(np.cross(plane_normal, tri[i] - tri[i - 1]).dot(p - tri[i]) >= 0 for i in range(3))


def ray_triangle_intersect(origin, r, tri):
    assert tri.shape == (3, 3)
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    n /= np.dot(n, n) ** 0.5
    r_dot_n = np.dot(r, n)
    if abs(r_dot_n) < 1e-8:
        print("r dot n too small", r_dot_n)
        return None
    t = np.dot(tri[0] - origin, n) / r_dot_n
    if t < 0:
        return None
    p = origin + r * t
    if point_in_triangle(p, tri):
        return p
    return None


def ray_quad_intersect(origin, r, quad):
    p = ray_triangle_intersect(origin, r, quad[:3])
    if p is not None:
        return p
    return ray_triangle_intersect(origin, r, quad[[2, 3, 0]])


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


def greedy_elevation_climb(elevations, start_r, start_c):
    r, c = start_r, start_c
    done = False
    k = 1
    while not done:
        print("climb", r, c, "elev", elevations[r, c])
        done = True
        for r2 in r + np.arange(-k, k+1, 1):
            if r2 < 0 or r2 >= elevations.shape[0]:
                continue
            for c2 in c + np.arange(-k, k+1, 1):
                if c2 < 0 or c2 >= elevations.shape[1]:
                    continue
                if elevations[r2, c2] > elevations[r, c]:
                    r, c = r2, c2
                    done = False
    return r, c


if __name__ == '__main__':
    data_dir = os.path.expanduser("~/Downloads/n40w107")


    def make_grids():
        grid_geo = make_grid_geo(data_dir)
        geo_ecef_converter = pygeodesy.ecef.EcefKarney(pygeodesy.Datums.WGS84)
        grid_ecef = np.zeros_like(grid_geo)  # x, y, z
        for r in range(grid_geo.shape[0]):
            for c in range(grid_geo.shape[1]):
                grid_ecef[r, c] = geo_ecef_converter.forward(*grid_geo[r, c])[:3]
        return grid_geo, grid_ecef


    grid_geo, grid_ecef = get_or_cache(data_dir, make_grids)

    print(grid_geo[0, 0], grid_geo[-1, -1])

    # ref_lat, ref_lon = 40.34663, -79.2263
    # ref_lat, ref_lon = 39.520719, -106.508642
    ref_lat, ref_lon = 39.515181, -106.506842
    ref_lat_lon = np.array([ref_lat, ref_lon])
    ref_idx = np.argmin(((grid_geo[:-1, :-1, :2] - ref_lat_lon) ** 2).sum(axis=-1))
    ref_r = ref_idx // (grid_geo.shape[0] - 1)
    ref_c = ref_idx % (grid_geo.shape[0] - 1)

    ref_r, ref_c = greedy_elevation_climb(grid_geo[:, :, 2], ref_r, ref_c)

    print(grid_geo[ref_r, ref_c, :2])
    ref_pt = grid_ecef[ref_r, ref_c]
    local_up = ref_pt / np.linalg.norm(ref_pt)
    local_north = np.array([0.0, 0.0, 1.0])
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

        MAX_DIST = 30000.0

        bfs = deque([(start_r, start_c)])
        visited[:, :] = False
        visited[start_r, start_c] = True
        view_area[start_r, start_c] = True

        quad = quad_at(start_r, start_c)
        hit_pos = ray_quad_intersect(origin, ray, quad)
        if hit_pos is not None:
            return np.linalg.norm(hit_pos - origin)

        while len(bfs) > 0:
            r1, c1 = bfs.popleft()
            for r2, c2 in [(r1 + 1, c1), (r1, c1 + 1), (r1 - 1, c1),
                           (r1, c1 - 1)]:
                if (r2 < 0 or r2 + 1 >= visited.shape[0] or c2 < 0
                        or c2 + 1 >= visited.shape[1] or visited[r2, c2]):
                    continue

                quad = quad_at(r2, c2)
                if ray_passes_over_quad(origin, ray, quad):
                    if np.linalg.norm(origin - quad.mean(axis=0)) > MAX_DIST:
                        return MAX_DIST
                    hit_pos = ray_quad_intersect(origin, ray, quad)
                    if hit_pos is not None:
                        return np.linalg.norm(hit_pos - origin)
                    bfs.append((r2, c2))
                visited[r2, c2] = True
                view_area[r2, c2] = True

        return None


    t0 = time.time()
    yaws = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    view_dists = []
    for yaw in yaws:
        view_dists.append(view_dist(ref_r, ref_c, yaw, -0.02, 2.0))
        print("yaw", yaw, "dist", view_dists[-1])
    if None in view_dists:
        print("view hit edge of loaded region")
    print("mean dist", np.mean(view_dists))
    print("time", time.time() - t0)
    print(ref_lat_lon)

    view_heights = grid_geo[:, :, 2] * view_area
    view_heights[view_area] -= np.min(view_heights[view_area])

    # plt.subplot(1, 2, 1)
    # plt.imshow(grid_geo[:, :, 2])
    # plt.subplot(1, 2, 2)
    plt.imshow(view_heights)
    plt.show()
