# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
import unittest

import numpy as np

import warp as wp
import warp.examples
from warp.tests.unittest_utils import *


@wp.kernel
def sample_mesh_query(
    mesh: wp.uint64,
    query_points: wp.array[wp.vec3],
    query_faces: wp.array[int],
    query_signs: wp.array[float],
    query_dist: wp.array[float],
):
    tid = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 10012.0

    p = query_points[tid]

    wp.mesh_query_point(mesh, p, max_dist, sign, face_index, face_u, face_v)

    cp = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    query_signs[tid] = sign
    query_faces[tid] = face_index
    query_dist[tid] = wp.length(cp - p)

    query = wp.mesh_query_point(mesh, p, max_dist)
    wp.expect_eq(query.sign, sign)
    wp.expect_eq(query.face, face_index)
    wp.expect_eq(query.u, face_u)
    wp.expect_eq(query.v, face_v)


@wp.kernel
def sample_mesh_query_no_sign(
    mesh: wp.uint64,
    query_points: wp.array[wp.vec3],
    query_faces: wp.array[int],
    query_dist: wp.array[float],
):
    tid = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)

    max_dist = 10012.0

    p = query_points[tid]

    wp.mesh_query_point_no_sign(mesh, p, max_dist, face_index, face_u, face_v)

    cp = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    query_faces[tid] = face_index
    query_dist[tid] = wp.length(cp - p)

    query = wp.mesh_query_point_no_sign(mesh, p, max_dist)
    wp.expect_eq(query.face, face_index)
    wp.expect_eq(query.u, face_u)
    wp.expect_eq(query.v, face_v)


@wp.kernel
def sample_mesh_query_sign_normal(
    mesh: wp.uint64,
    query_points: wp.array[wp.vec3],
    query_faces: wp.array[int],
    query_signs: wp.array[float],
    query_dist: wp.array[float],
):
    tid = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 10012.0

    p = query_points[tid]

    wp.mesh_query_point_sign_normal(mesh, p, max_dist, sign, face_index, face_u, face_v)

    cp = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    query_signs[tid] = sign
    query_faces[tid] = face_index
    query_dist[tid] = wp.length(cp - p)

    query = wp.mesh_query_point_sign_normal(mesh, p, max_dist)
    wp.expect_eq(query.sign, sign)
    wp.expect_eq(query.face, face_index)
    wp.expect_eq(query.u, face_u)
    wp.expect_eq(query.v, face_v)


@wp.kernel
def sample_mesh_query_sign_winding_number(
    mesh: wp.uint64,
    query_points: wp.array[wp.vec3],
    query_faces: wp.array[int],
    query_signs: wp.array[float],
    query_dist: wp.array[float],
):
    tid = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 10012.0

    p = query_points[tid]

    wp.mesh_query_point_sign_winding_number(mesh, p, max_dist, sign, face_index, face_u, face_v)

    cp = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    query_signs[tid] = sign
    query_faces[tid] = face_index
    query_dist[tid] = wp.length(cp - p)

    query = wp.mesh_query_point_sign_winding_number(mesh, p, max_dist)
    wp.expect_eq(query.sign, sign)
    wp.expect_eq(query.face, face_index)
    wp.expect_eq(query.u, face_u)
    wp.expect_eq(query.v, face_v)


@wp.kernel
def sample_mesh_query_sign_parity(
    mesh: wp.uint64,
    query_points: wp.array[wp.vec3],
    query_faces: wp.array[int],
    query_signs: wp.array[float],
    query_dist: wp.array[float],
):
    tid = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 10012.0

    p = query_points[tid]

    wp.mesh_query_point_sign_parity(mesh, p, max_dist, sign, face_index, face_u, face_v)

    cp = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    query_signs[tid] = sign
    query_faces[tid] = face_index
    query_dist[tid] = wp.length(cp - p)

    query = wp.mesh_query_point_sign_parity(mesh, p, max_dist)
    wp.expect_eq(query.sign, sign)
    wp.expect_eq(query.face, face_index)
    wp.expect_eq(query.u, face_u)
    wp.expect_eq(query.v, face_v)


@wp.func
def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec2(1.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if d3 >= 0.0 and d4 <= d3:
        return wp.vec2(0.0, 1.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return wp.vec2(1.0 - v, v)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    if d6 >= 0.0 and d5 <= d6:
        return wp.vec2(0.0, 0.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return wp.vec2(1.0 - w, 0.0)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return wp.vec2(0.0, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w

    return wp.vec2(u, v)


@wp.func
def solid_angle(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, p: wp.vec3):
    a = v0 - p
    b = v1 - p
    c = v2 - p

    a_len = wp.length(a)
    b_len = wp.length(b)
    c_len = wp.length(c)

    det = wp.dot(a, wp.cross(b, c))
    den = a_len * b_len * c_len + wp.dot(a, b) * c_len + wp.dot(b, c) * a_len + wp.dot(c, a) * b_len

    return 2.0 * wp.atan2(det, den)


@wp.kernel
def sample_mesh_brute(
    tri_points: wp.array[wp.vec3],
    tri_indices: wp.array[int],
    tri_count: int,
    query_points: wp.array[wp.vec3],
    query_faces: wp.array[int],
    query_signs: wp.array[float],
    query_dist: wp.array[float],
):
    tid = wp.tid()

    min_face = int(0)
    min_dist = float(1.0e6)

    sum_solid_angle = float(0.0)

    p = query_points[tid]

    for i in range(0, tri_count):
        a = tri_points[tri_indices[i * 3 + 0]]
        b = tri_points[tri_indices[i * 3 + 1]]
        c = tri_points[tri_indices[i * 3 + 2]]

        sum_solid_angle += solid_angle(a, b, c, p)

        bary = triangle_closest_point(a, b, c, p)
        u = bary[0]
        v = bary[1]

        cp = u * a + v * b + (1.0 - u - v) * c
        cp_dist = wp.length(cp - p)

        if cp_dist < min_dist:
            min_dist = cp_dist
            min_face = i

    # for an inside point, the sum of the solid angle should be 4PI
    # for an outside point, the sum should be 0
    query_faces[tid] = min_face
    query_signs[tid] = sum_solid_angle
    query_dist[tid] = min_dist


# constructs a grid of evenly spaced particles
def particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter):
    rng = np.random.default_rng(123)

    points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
    points_t = np.array((points[0], points[1], points[2])).T * radius * 2.0 + np.array(lower)
    points_t = points_t + rng.random(points_t.shape) * radius * jitter

    return points_t.reshape((-1, 3))


# triangulate a list of polygon face indices
def triangulate(face_counts, face_indices):
    num_tris = np.sum(np.subtract(face_counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros(num_tri_vtx, dtype=int)
    ctr = 0
    wedgeIdx = 0

    for nb in face_counts:
        for i in range(nb - 2):
            tri_indices[ctr] = face_indices[wedgeIdx]
            tri_indices[ctr + 1] = face_indices[wedgeIdx + i + 1]
            tri_indices[ctr + 2] = face_indices[wedgeIdx + i + 2]
            ctr += 3
        wedgeIdx += nb

    return tri_indices


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_mesh_query_point(test, device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    mesh = Usd.Stage.Open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "spiky.usd")))
    mesh_geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/Cube/Cube"))

    mesh_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
    mesh_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

    tri_indices = triangulate(mesh_counts, mesh_indices)

    mesh_points = wp.array(np.array(mesh_geom.GetPointsAttr().Get()), dtype=wp.vec3, device=device)
    mesh_indices = wp.array(np.array(tri_indices), dtype=int, device=device)

    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    if wp.is_cubql_available():
        constructors.append("cubql")

    leaf_sizes = [1, 2, 4]

    for leaf_size, constructor in itertools.product(leaf_sizes, constructors):
        # cuBQL doesn't yet support winding-number queries; build without that data
        # and skip the winding-number kernel / assertions for this constructor.
        use_winding_number = constructor != "cubql"

        # create mesh
        mesh = wp.Mesh(
            points=mesh_points,
            velocities=None,
            indices=mesh_indices,
            support_winding_number=use_winding_number,
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )

        p = particle_grid(32, 32, 32, np.array([-1.1, -1.1, -1.1]), 0.05, 0.0)

        query_count = len(p)
        query_points = wp.array(p, dtype=wp.vec3, device=device)

        signs_query = wp.zeros(query_count, dtype=float, device=device)
        faces_query = wp.zeros(query_count, dtype=int, device=device)
        dist_query = wp.zeros(query_count, dtype=float, device=device)

        faces_query_no_sign = wp.zeros(query_count, dtype=int, device=device)
        dist_query_no_sign = wp.zeros(query_count, dtype=float, device=device)

        signs_query_normal = wp.zeros(query_count, dtype=float, device=device)
        faces_query_normal = wp.zeros(query_count, dtype=int, device=device)
        dist_query_normal = wp.zeros(query_count, dtype=float, device=device)

        signs_query_winding_number = wp.zeros(query_count, dtype=float, device=device)
        faces_query_winding_number = wp.zeros(query_count, dtype=int, device=device)
        dist_query_winding_number = wp.zeros(query_count, dtype=float, device=device)

        signs_query_parity = wp.zeros(query_count, dtype=float, device=device)
        faces_query_parity = wp.zeros(query_count, dtype=int, device=device)
        dist_query_parity = wp.zeros(query_count, dtype=float, device=device)

        signs_brute = wp.zeros(query_count, dtype=float, device=device)
        faces_brute = wp.zeros(query_count, dtype=int, device=device)
        dist_brute = wp.zeros(query_count, dtype=float, device=device)

        wp.launch(
            kernel=sample_mesh_query,
            dim=query_count,
            inputs=[mesh.id, query_points, faces_query, signs_query, dist_query],
            device=device,
        )

        wp.launch(
            kernel=sample_mesh_query_no_sign,
            dim=query_count,
            inputs=[mesh.id, query_points, faces_query_no_sign, dist_query_no_sign],
            device=device,
        )

        wp.launch(
            kernel=sample_mesh_query_sign_normal,
            dim=query_count,
            inputs=[mesh.id, query_points, faces_query_normal, signs_query_normal, dist_query_normal],
            device=device,
        )

        if use_winding_number:
            wp.launch(
                kernel=sample_mesh_query_sign_winding_number,
                dim=query_count,
                inputs=[
                    mesh.id,
                    query_points,
                    faces_query_winding_number,
                    signs_query_winding_number,
                    dist_query_winding_number,
                ],
                device=device,
            )

        wp.launch(
            kernel=sample_mesh_query_sign_parity,
            dim=query_count,
            inputs=[
                mesh.id,
                query_points,
                faces_query_parity,
                signs_query_parity,
                dist_query_parity,
            ],
            device=device,
        )

        wp.launch(
            kernel=sample_mesh_brute,
            dim=query_count,
            inputs=[
                mesh_points,
                mesh_indices,
                int(len(mesh_indices) / 3),
                query_points,
                faces_brute,
                signs_brute,
                dist_brute,
            ],
            device=device,
        )

        signs_query = signs_query.numpy()
        faces_query = faces_query.numpy()
        dist_query = dist_query.numpy()

        faces_query_no_sign = faces_query_no_sign.numpy()
        dist_query_no_sign = dist_query_no_sign.numpy()

        signs_query_normal = signs_query_normal.numpy()
        faces_query_normal = faces_query_normal.numpy()
        dist_query_normal = dist_query_normal.numpy()

        signs_query_winding_number = signs_query_winding_number.numpy()
        faces_query_winding_number = faces_query_winding_number.numpy()
        dist_query_winding_number = dist_query_winding_number.numpy()

        signs_query_parity = signs_query_parity.numpy()
        faces_query_parity = faces_query_parity.numpy()
        dist_query_parity = dist_query_parity.numpy()

        signs_brute = signs_brute.numpy()
        faces_brute = faces_brute.numpy()
        dist_brute = dist_brute.numpy()

        query_points = query_points.numpy()

        inside_query = [[0.0, 0.0, 0.0]]
        inside_query_normal = [[0.0, 0.0, 0.0]]
        inside_query_winding_number = [[0.0, 0.0, 0.0]]
        inside_query_parity = [[0.0, 0.0, 0.0]]
        inside_brute = [[0.0, 0.0, 0.0]]

        for i in range(query_count):
            if signs_query[i] < 0.0:
                inside_query.append(query_points[i].tolist())

            if signs_query_normal[i] < 0.0:
                inside_query_normal.append(query_points[i].tolist())

            if signs_query_winding_number[i] < 0.0:
                inside_query_winding_number.append(query_points[i].tolist())

            if signs_query_parity[i] < 0.0:
                inside_query_parity.append(query_points[i].tolist())

            if signs_brute[i] > math.pi * 2.0:
                inside_brute.append(query_points[i].tolist())

        inside_query = np.array(inside_query)
        inside_query_normal = np.array(inside_query_normal)
        inside_query_winding_number = np.array(inside_query_winding_number)
        inside_query_parity = np.array(inside_query_parity)
        inside_brute = np.array(inside_brute)

        # import warp.render

        # stage = warp.render.UsdRenderer("tests/outputs/test_mesh_query_point.usd")

        # radius = 0.1
        # stage.begin_frame(0.0)
        # stage.render_mesh(points=mesh_points.numpy(), indices=mesh_indices.numpy(), name="mesh")
        # stage.render_points(points=inside_query, radius=radius, name="query")
        # stage.render_points(points=inside_brute, radius=radius, name="brute")
        # stage.render_points(points=query_points, radius=radius, name="all")
        # stage.end_frame()

        # stage.save()

        test.assertTrue(len(inside_query) == len(inside_brute))
        test.assertTrue(len(inside_query_normal) == len(inside_brute))
        if use_winding_number:
            test.assertTrue(len(inside_query_winding_number) == len(inside_brute))
        test.assertTrue(len(inside_query_parity) == len(inside_brute))

        tolerance = 1.5e-4
        dist_error = np.max(np.abs(dist_query - dist_brute))
        sign_error = np.max(np.abs(inside_query - inside_brute))

        test.assertTrue(dist_error < tolerance, f"mesh_query_point dist_error is {dist_error} which is >= {tolerance}")
        test.assertTrue(sign_error < tolerance, f"mesh_query_point sign_error is {sign_error} which is >= {tolerance}")

        dist_error = np.max(np.abs(dist_query_no_sign - dist_brute))

        test.assertTrue(
            dist_error < tolerance, f"mesh_query_point_no_sign dist_error is {dist_error} which is >= {tolerance}"
        )

        dist_error = np.max(np.abs(dist_query_normal - dist_brute))
        sign_error = np.max(np.abs(inside_query_normal - inside_brute))

        test.assertTrue(
            dist_error < tolerance, f"mesh_query_point_sign_normal dist_error is {dist_error} which is >= {tolerance}"
        )
        test.assertTrue(
            sign_error < tolerance, f"mesh_query_point_sign_normal sign_error is {sign_error} which is >= {tolerance}"
        )

        if use_winding_number:
            dist_error = np.max(np.abs(dist_query_winding_number - dist_brute))
            sign_error = np.max(np.abs(inside_query_winding_number - inside_brute))

            test.assertTrue(
                dist_error < tolerance,
                f"mesh_query_point_sign_winding_number dist_error is {dist_error} which is >= {tolerance}",
            )
            test.assertTrue(
                sign_error < tolerance,
                f"mesh_query_point_sign_winding_number sign_error is {sign_error} which is >= {tolerance}",
            )

        dist_error = np.max(np.abs(dist_query_parity - dist_brute))
        sign_error = np.max(np.abs(inside_query_parity - inside_brute))

        test.assertTrue(
            dist_error < tolerance,
            f"mesh_query_point_sign_parity dist_error is {dist_error} which is >= {tolerance}",
        )
        test.assertTrue(
            sign_error < tolerance,
            f"mesh_query_point_sign_parity sign_error is {sign_error} which is >= {tolerance}",
        )


@wp.kernel
def mesh_query_point_loss(
    mesh: wp.uint64,
    query_points: wp.array[wp.vec3],
    projected_points: wp.array[wp.vec3],
    loss: wp.array[float],
):
    tid = wp.tid()

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    max_dist = 10012.0

    p = query_points[tid]

    wp.mesh_query_point(mesh, p, max_dist, sign, face_index, face_u, face_v)
    q = wp.mesh_eval_position(mesh, face_index, face_u, face_v)

    projected_points[tid] = q

    dist = wp.length(wp.sub(p, q))
    loss[tid] = dist


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_adj_mesh_query_point(test, device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    mesh = Usd.Stage.Open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "torus.usda")))
    mesh_geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/World/Torus"))

    mesh_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
    mesh_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

    tri_indices = triangulate(mesh_counts, mesh_indices)

    mesh_points = wp.array(np.array(mesh_geom.GetPointsAttr().Get()), dtype=wp.vec3, device=device)
    mesh_indices = wp.array(np.array(tri_indices), dtype=int, device=device)

    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    for constructor in constructors:
        # test tri
        # print("Testing Single Triangle")
        # mesh_points = wp.array(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), dtype=wp.vec3, device=device)
        # mesh_indices = wp.array(np.array([0,1,2]), dtype=int, device=device)

        # create mesh
        mesh = wp.Mesh(
            points=mesh_points,
            velocities=None,
            indices=mesh_indices,
            support_winding_number=True,
            bvh_constructor=constructor,
        )

        # p = particle_grid(32, 32, 32, np.array([-5.0, -5.0, -5.0]), 0.1, 0.1)*100.0
        p = wp.vec3(50.0, 50.0, 50.0)

        tape = wp.Tape()

        # analytic gradients
        with tape:
            query_points = wp.array(p, dtype=wp.vec3, device=device, requires_grad=True)
            projected_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
            loss = wp.zeros(n=1, dtype=float, device=device, requires_grad=True)

            wp.launch(
                kernel=mesh_query_point_loss,
                dim=1,
                inputs=[mesh.id, query_points, projected_points, loss],
                device=device,
            )

        tape.backward(loss=loss)
        analytic = tape.gradients[query_points].numpy().flatten()

        # numeric gradients
        eps = 1.0e-3
        loss_values = []
        numeric = np.zeros(3)

        offset_query_points = [
            wp.vec3(p[0] - eps, p[1], p[2]),
            wp.vec3(p[0] + eps, p[1], p[2]),
            wp.vec3(p[0], p[1] - eps, p[2]),
            wp.vec3(p[0], p[1] + eps, p[2]),
            wp.vec3(p[0], p[1], p[2] - eps),
            wp.vec3(p[0], p[1], p[2] + eps),
        ]

        for i in range(6):
            q = offset_query_points[i]

            query_points = wp.array(q, dtype=wp.vec3, device=device)
            projected_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
            loss = wp.zeros(n=1, dtype=float, device=device)

            wp.launch(
                kernel=mesh_query_point_loss,
                dim=1,
                inputs=[mesh.id, query_points, projected_points, loss],
                device=device,
            )

            loss_values.append(loss.numpy()[0])

        for i in range(3):
            l_0 = loss_values[i * 2]
            l_1 = loss_values[i * 2 + 1]
            gradient = (l_1 - l_0) / (2.0 * eps)
            numeric[i] = gradient

        error = ((analytic - numeric) * (analytic - numeric)).sum(axis=0)

        tolerance = 1.0e-3
        test.assertTrue(error < tolerance, f"error is {error} which is >= {tolerance}")


@wp.kernel
def sample_furthest_points(mesh: wp.uint64, query_points: wp.array[wp.vec3], query_result: wp.array[float]):
    tid = wp.tid()

    p = query_points[tid]

    face = int(0)
    bary_u = float(0.0)
    bary_v = float(0.0)

    if wp.mesh_query_furthest_point_no_sign(mesh, p, 0.0, face, bary_u, bary_v):
        closest = wp.mesh_eval_position(mesh, face, bary_u, bary_v)

        query_result[tid] = wp.length_sq(p - closest)

    query = wp.mesh_query_furthest_point_no_sign(mesh, p, 0.0)
    wp.expect_eq(query.face, face)
    wp.expect_eq(query.u, bary_u)
    wp.expect_eq(query.v, bary_v)


@wp.kernel
def sample_furthest_points_brute(
    mesh_points: wp.array[wp.vec3], query_points: wp.array[wp.vec3], query_result: wp.array[float]
):
    tid = wp.tid()

    p = query_points[tid]
    max_dist_sq = float(0.0)

    for i in range(mesh_points.shape[0]):
        dist_sq = wp.length_sq(p - mesh_points[i])

        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq

    query_result[tid] = max_dist_sq


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_mesh_query_furthest_point(test, device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    mesh = Usd.Stage.Open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "spiky.usd")))
    mesh_geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/Cube/Cube"))

    mesh_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
    mesh_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

    tri_indices = triangulate(mesh_counts, mesh_indices)

    mesh_points = wp.array(np.array(mesh_geom.GetPointsAttr().Get()), dtype=wp.vec3, device=device)
    mesh_indices = wp.array(np.array(tri_indices), dtype=int, device=device)

    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    if wp.is_cubql_available():
        constructors.append("cubql")

    for constructor in constructors:
        # create mesh
        mesh = wp.Mesh(points=mesh_points, indices=mesh_indices, bvh_constructor=constructor)

        p = particle_grid(32, 32, 32, np.array([-1.1, -1.1, -1.1]), 0.05, 0.0)

        query_count = len(p)
        query_points = wp.array(p, dtype=wp.vec3, device=device)

        dist_query = wp.zeros(query_count, dtype=float, device=device)
        dist_brute = wp.zeros(query_count, dtype=float, device=device)

        wp.launch(sample_furthest_points, dim=query_count, inputs=[mesh.id, query_points, dist_query], device=device)
        wp.launch(
            sample_furthest_points_brute, dim=query_count, inputs=[mesh_points, query_points, dist_brute], device=device
        )

        assert_np_equal(dist_query.numpy(), dist_brute.numpy(), tol=1.0e-3)


@wp.func
def triangle_closest_point_for_test(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        bary = wp.vec3(1.0, 0.0, 0.0)
        return a, bary

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        bary = wp.vec3(0.0, 1.0, 0.0)
        return b, bary

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        bary = wp.vec3(0.0, 0.0, 1.0)
        return c, bary

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        bary = wp.vec3(1.0 - v, v, 0.0)
        return a + v * ab, bary

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        bary = wp.vec3(1.0 - v, 0.0, v)
        return a + v * ac, bary

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bary = wp.vec3(0.0, 1.0 - v, v)
        return b + v * (c - b), bary

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    bary = wp.vec3(1.0 - v - w, v, w)
    return a + v * ab + w * ac, bary


def load_mesh(model_name="bunny"):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    usd_stage = Usd.Stage.Open(os.path.join(wp.examples.get_asset_directory(), model_name + ".usd"))
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/" + model_name))

    vertices = np.array(usd_geom.GetPointsAttr().Get())
    faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

    return vertices, faces


@wp.kernel
def point_query_aabb_and_closest(
    query_radius: float,
    mesh_id: wp.uint64,
    pts: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    query_results_num_collisions: wp.array[wp.int32],
    query_results_min_dist: wp.array[float],
    query_results_closest_point_velocity: wp.array[wp.vec3],
):
    p_index = wp.tid()
    p = pts[p_index]

    lower = wp.vec3(p[0] - query_radius, p[1] - query_radius, p[2] - query_radius)
    upper = wp.vec3(p[0] + query_radius, p[1] + query_radius, p[2] + query_radius)

    closest_query = wp.mesh_query_point_no_sign(mesh_id, p, query_radius)
    if closest_query.result:
        closest_p = wp.mesh_eval_position(mesh_id, closest_query.face, closest_query.u, closest_query.v)
        closest_p_vel = wp.mesh_eval_velocity(mesh_id, closest_query.face, closest_query.u, closest_query.v)

        query_results_min_dist[p_index] = wp.length(closest_p - p)
        query_results_closest_point_velocity[p_index] = closest_p_vel

    query = wp.mesh_query_aabb(mesh_id, lower, upper)

    tri_index = wp.int32(0)
    num_collisions = wp.int32(0)
    while wp.mesh_query_aabb_next(query, tri_index):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]

        u1 = pos[t1]
        u2 = pos[t2]
        u3 = pos[t3]

        closest_p1, _barycentric1 = triangle_closest_point_for_test(u1, u2, u3, p)

        dis = wp.length(closest_p1 - p)

        if dis < query_radius:
            num_collisions = num_collisions + 1

    query_results_num_collisions[p_index] = num_collisions


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_set_mesh_points(test, device):
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    rng = np.random.default_rng(123)

    models = ["bunny", "nonuniform"]

    for model in models:
        vs, fs = load_mesh(model)
        vertices1 = wp.array(vs, dtype=wp.vec3, device=device)
        velocities1_np = rng.standard_normal(size=(vertices1.shape[0], 3))
        velocities1 = wp.array(velocities1_np, dtype=wp.vec3, device=device)

        faces = wp.array(fs, dtype=wp.int32, device=device)

        n = 1000
        query_radius = 0.2
        pts1 = wp.array(rng.standard_normal(size=(n, 3)), dtype=wp.vec3, device=device)

        query_results_num_cols1 = wp.zeros(n, dtype=wp.int32, device=device)
        query_results_min_dist1 = wp.zeros(n, dtype=float, device=device)
        query_results_closest_point_velocity1 = wp.zeros(n, dtype=wp.vec3, device=device)

        for constructor in constructors:
            mesh = wp.Mesh(vertices1, faces, velocities=velocities1, bvh_constructor=constructor)
            fs_2D = faces.reshape((-1, 3))

            wp.launch(
                kernel=point_query_aabb_and_closest,
                inputs=[
                    query_radius,
                    mesh.id,
                    pts1,
                    vertices1,
                    fs_2D,
                    query_results_num_cols1,
                    query_results_min_dist1,
                    query_results_closest_point_velocity1,
                ],
                dim=n,
                device=device,
            )

            shift = rng.standard_normal(size=3)

            vs_higher = vs + shift
            vertices2 = wp.array(vs_higher, dtype=wp.vec3, device=device)

            velocities2_np = velocities1_np + shift[None, ...]
            velocities2 = wp.array(velocities2_np, dtype=wp.vec3, device=device)

            pts2 = wp.array(pts1.numpy() + shift, dtype=wp.vec3, device=device)

            mesh.points = vertices2
            mesh.velocities = velocities2

            query_results_num_cols2 = wp.zeros(n, dtype=wp.int32, device=device)
            query_results_min_dist2 = wp.zeros(n, dtype=float, device=device)
            query_results_closest_point_velocity2 = wp.array([shift for i in range(n)], dtype=wp.vec3, device=device)

            wp.launch(
                kernel=point_query_aabb_and_closest,
                inputs=[
                    query_radius,
                    mesh.id,
                    pts2,
                    vertices2,
                    fs_2D,
                    query_results_num_cols2,
                    query_results_min_dist2,
                    query_results_closest_point_velocity2,
                ],
                dim=n,
                device=device,
            )

            test.assertTrue((query_results_num_cols1.numpy() == query_results_num_cols2.numpy()).all())
            test.assertTrue(((query_results_min_dist1.numpy() - query_results_min_dist2.numpy()) < 1e-5).all())


@wp.kernel
def sample_parity_bounds(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    n_sample: int,
    max_dist: float,
    results: wp.array2d[float],
):
    i = wp.tid()
    p = points[i]
    query = wp.mesh_query_point_sign_parity(mesh, p, max_dist, n_sample)
    nearest = wp.mesh_query_point_no_sign(mesh, p, max_dist)
    sign = float(0.0)
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    found = wp.mesh_query_point_sign_parity(mesh, p, max_dist, sign, face, u, v, n_sample)
    wp.expect_eq(query.result, nearest.result)
    wp.expect_eq(query.result, found)
    results[i, 0] = float(query.result)
    if query.result:
        wp.expect_eq(query.face, nearest.face)
        wp.expect_eq(query.u, nearest.u)
        wp.expect_eq(query.v, nearest.v)
        wp.expect_eq(query.sign, sign)
        wp.expect_eq(query.face, face)
        wp.expect_eq(query.u, u)
        wp.expect_eq(query.v, v)
        if n_sample == 1:
            default_query = wp.mesh_query_point_sign_parity(mesh, p, max_dist)
            wp.expect_eq(query.sign, default_query.sign)
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        results[i, 1] = query.sign
        results[i, 2] = closest[0]
        results[i, 3] = closest[1]
        results[i, 4] = closest[2]


def parity_tetrahedron(device, constructor="sah", leaf_size=4):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.ravel(), dtype=wp.int32, device=device),
        bvh_constructor=constructor,
        bvh_leaf_size=leaf_size,
    )


def parity_bounds_constructors(device):
    constructors = ["sah", "median"] if device.is_cpu else ["sah", "median", "lbvh"]
    if wp.is_cubql_available():
        constructors.append("cubql")
    return constructors


def parity_bounds_results(mesh, points, n_sample=1, max_dist=10.0):
    points = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=mesh.device)
    results = wp.zeros((len(points), 5), dtype=float, device=mesh.device)
    wp.launch(
        sample_parity_bounds,
        dim=len(points),
        inputs=[mesh.id, points, n_sample, max_dist],
        outputs=[results],
        device=mesh.device,
    )
    return results.numpy()


def test_mesh_query_parity_outside_bounds(test, device):
    # The first two exterior points were misclassified by the default single
    # ray on CPU and CUDA respectively; rounding differs between backends.
    # The other exterior points cover all six AABB sides. The last two points
    # distinguish solid containment from containment in the bounding box.
    points = [
        [-0.0037126243114471436, -0.1358090341091156, -0.1293368637561798],
        [-0.5148504972457886, -0.0432361364364624, -0.5173474550247192],
        [-0.25, 0.2, 0.3],
        [1.25, 0.2, 0.3],
        [0.2, -0.25, 0.3],
        [0.2, 1.25, 0.3],
        [0.2, 0.3, -0.25],
        [0.2, 0.3, 1.25],
        [0.2, 0.2, 0.2],
        [0.6, 0.6, 0.6],
    ]
    for constructor, leaf_size in itertools.product(parity_bounds_constructors(device), [1, 4]):
        with test.subTest(constructor=constructor, leaf_size=leaf_size):
            mesh = parity_tetrahedron(device, constructor, leaf_size)
            results = parity_bounds_results(mesh, points)
            np.testing.assert_array_equal(results[:, 0], np.ones(len(points)))
            np.testing.assert_array_equal(results[:, 1], [1, 1, 1, 1, 1, 1, 1, 1, -1, 1])
            np.testing.assert_array_equal(results[:2, 2:], np.zeros((2, 3)))


def test_mesh_query_parity_bounds_sample_count(test, device):
    mesh = parity_tetrahedron(device)
    points = [[-0.25, 0.2, 0.3], [0.2, 0.2, 0.2], [0.6, 0.6, 0.6]]
    for n_sample in [-1, 0, 1, 2, 3, 5]:
        with test.subTest(n_sample=n_sample):
            results = parity_bounds_results(mesh, points, n_sample=n_sample)
            np.testing.assert_array_equal(results[:, 0], [1, 1, 1])
            np.testing.assert_array_equal(results[:, 1], [-1, -1, -1] if n_sample <= 0 else [1, -1, 1])
    # Bounds-based sign classification must not turn a distance-limited miss
    # into a successful nearest-point query.
    results = parity_bounds_results(mesh, points[:1], max_dist=0.1)
    test.assertEqual(results[0, 0], 0)


def test_mesh_query_parity_bounds_refit(test, device):
    point = [[-0.25, -0.25, -0.25]]

    def check_sign(mesh, expected):
        result = parity_bounds_results(mesh, point)
        test.assertEqual(result[0, 0], 1)
        test.assertEqual(result[0, 1], expected)

    for constructor in parity_bounds_constructors(device):
        with test.subTest(constructor=constructor):
            mesh = parity_tetrahedron(device, constructor)
            original = mesh.points.numpy().copy()
            expanded = 4.0 * original - 1.0
            check_sign(mesh, 1)
            # The point lies outside the original AABB but inside the expanded solid.
            mesh.points.assign(expanded)
            mesh.refit()
            check_sign(mesh, -1)
            # Assigning a new points array automatically refits the BVH.
            mesh.points = wp.array(expanded + 4.0, dtype=wp.vec3, device=device)
            check_sign(mesh, 1)
            mesh.points = wp.array(expanded, dtype=wp.vec3, device=device)
            check_sign(mesh, -1)
            mesh.points.assign(original)
            mesh.refit()
            check_sign(mesh, 1)


# A pair of adjacent faces reduced from a close-clearance mesh query. Keeping
# the original binary32 coordinates makes this independent of any CAD asset.
def closest_point_precision_fixture():
    vertices = np.array(
        [
            [0.0197, 0.0124, -0.0144],
            [0.0197, 0.0124, 0.0144],
            [0.0227, 0.0094, -0.0144],
            [0.0191, 0.0124, -0.0144],
        ],
        dtype=np.float32,
    )
    points = np.array(
        [
            [0.019699811935424805, 0.012403983622789383, 0.003116013016551733],
            [0.019699353724718094, 0.012403230182826519, -0.0031294734217226505],
        ],
        dtype=np.float32,
    )
    return vertices, np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32), points


def closest_point_boundary_fixture():
    # Refining the face projection must not return coordinates outside the
    # triangle. These two points project onto AC and BC respectively.
    data = np.array(
        [
            [
                [1109193749, 1109307421, 3254786347],
                [3289244216, 1144799206, 1134515891],
                [3284410732, 1141405764, 1129860094],
                [3274738941, 1133665002, 1119381205],
            ],
            [
                [3106103682, 3110716876, 3141419707],
                [3169175261, 999262576, 1005180076],
                [3164833088, 994544879, 998922122],
                [3167004306, 997400312, 1002052511],
            ],
        ],
        dtype=np.uint32,
    ).view(np.float32)
    return data[:, :3], data[:, 3]


def closest_point_triangle_reference(triangle, point):
    """Independent float64 finite-triangle projection, including all edges."""
    a, b, c = np.asarray(triangle, dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    candidates = []
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length_sq = np.dot(edge, edge)
        t = np.clip(np.dot(p - start, edge) / length_sq, 0.0, 1.0) if length_sq else 0.0
        candidates.append(start + t * edge)
    normal = np.cross(b - a, c - a)
    normal_sq = np.dot(normal, normal)
    if normal_sq:
        projection = p - normal * (np.dot(p - a, normal) / normal_sq)
        if all(
            np.dot(np.cross(end - start, projection - start), normal) >= 0 for start, end in ((a, b), (b, c), (c, a))
        ):
            candidates.append(projection)
    return min(candidates, key=lambda q: np.dot(p - q, p - q))


@wp.func_native(
    "return wp::closest_point_to_triangle(a, b, c, p);",
    adj_snippet="wp::adj_closest_point_to_triangle(a, b, c, p, adj_a, adj_b, adj_c, adj_p, adj_ret);",
)
def closest_point_precision_native(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3) -> wp.vec2: ...


@wp.kernel
def closest_point_precision_kernel(
    triangles: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    closest: wp.array[wp.vec3],
    barycentric: wp.array[wp.vec2],
):
    i = wp.tid()
    a = triangles[3 * i]
    b = triangles[3 * i + 1]
    c = triangles[3 * i + 2]
    uv = closest_point_precision_native(a, b, c, points[i])
    closest[i] = uv[0] * a + uv[1] * b + (1.0 - uv[0] - uv[1]) * c
    barycentric[i] = uv


@wp.kernel(enable_backward=False)
def closest_point_precision_mesh_kernel(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    closest: wp.array[wp.vec3],
    faces: wp.array[int],
):
    i = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh, points[i], 1.0)
    parity = wp.mesh_query_point_sign_parity(mesh, points[i], 1.0)
    wp.expect_eq(query.result, True)
    wp.expect_eq(parity.result, True)
    wp.expect_eq(query.face, parity.face)
    wp.expect_eq(query.u, parity.u)
    wp.expect_eq(query.v, parity.v)
    closest[i] = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
    faces[i] = query.face


def test_closest_point_triangle_precision(test, device):
    vertices, indices, points = closest_point_precision_fixture()
    triangle = vertices[indices[1]]
    rng = np.random.default_rng(14892)
    nearby = np.column_stack(
        (
            triangle[0, 0] + rng.uniform(-2e-6, 2e-6, 32),
            triangle[0, 1] + rng.uniform(0.5e-6, 8e-6, 32),
            rng.uniform(-0.01, 0.01, 32),
        )
    ).astype(np.float32)
    points = np.concatenate((points, nearby))
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    frames = (
        (np.eye(3), np.zeros(3)),
        (np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), np.array([0.00390625, -0.0078125, 0.015625])),
        (rotation, np.array([-0.0078125, 0.00390625, 0])),
    )
    triangles, queries = [], []
    for permutation, (rotation, translation) in itertools.product(itertools.permutations(range(3)), frames):
        tri = (triangle[list(permutation)].astype(np.float64) @ rotation.T + translation).astype(np.float32)
        q = (points.astype(np.float64) @ rotation.T + translation).astype(np.float32)
        triangles.extend([tri] * len(q))
        queries.extend(q)
    # Exercise every ordinary vertex and edge region as well as the face.
    regular = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    ordinary = [
        [-0.2, -0.1, 0.3],
        [1.2, -0.1, 0.3],
        [-0.1, 1.2, 0.3],
        [0.3, -0.1, 0.3],
        [-0.1, 0.3, 0.3],
        [0.7, 0.7, 0.3],
        [0.2, 0.3, 0.3],
    ]
    triangles.extend([regular] * len(ordinary))
    queries.extend(ordinary)
    # Preserve defined coincident/collinear cases without introducing a new
    # contract for arbitrary degenerate meshes or differentiating boundaries.
    triangles.extend([[[0.1, 0.2, 0.3]] * 3, [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0], [1, 0, 0], [2, 0, 0]]])
    queries.extend([[0.3, 0.3, 0.1], [0.5, 0.2, 0], [1.5, 0.2, 0]])
    # Rotated skinny faces must also satisfy the face projection equations.
    # These bit-exact cases cover aspect ratios 1e-3, 1e-4 and 1e-5.
    skinny = np.array(
        [
            [
                [3182349133, 1025857097, 3171818777],
                [3207811024, 1057218383, 1058555146],
                [3204808973, 1052842367, 1054046086],
                [3197850419, 1045813758, 1044492726],
            ],
            [
                [942711382, 3090412363, 948933577],
                [3121259607, 962874410, 3124371722],
                [3116264975, 957678763, 3120426302],
                [3119112246, 960270185, 3122433062],
            ],
            [
                [1017209618, 1036078285, 3182284678],
                [3208203339, 3184204214, 1057891044],
                [3204585784, 3174061091, 1052588896],
                [3199481345, 3139168957, 1047585051],
            ],
        ],
        dtype=np.uint32,
    ).view(np.float32)
    triangles.extend(skinny[:, :3])
    queries.extend(skinny[:, 3])
    boundary_triangles, boundary_points = closest_point_boundary_fixture()
    triangles.extend(boundary_triangles)
    queries.extend(boundary_points)
    triangles, queries = np.asarray(triangles, np.float32), np.asarray(queries, np.float32)
    expected = np.array([closest_point_triangle_reference(t, p) for t, p in zip(triangles, queries, strict=True)])
    closest = wp.empty(len(queries), dtype=wp.vec3, device=device)
    barycentric = wp.empty(len(queries), dtype=wp.vec2, device=device)
    wp.launch(
        closest_point_precision_kernel,
        dim=len(queries),
        inputs=[
            wp.array(triangles.reshape(-1, 3), dtype=wp.vec3, device=device),
            wp.array(queries, dtype=wp.vec3, device=device),
        ],
        outputs=[closest, barycentric],
        device=device,
    )
    result = closest.numpy().astype(np.float64)
    # Allow the final binary32 barycentric reconstruction, not the much larger
    # cancellation error in the region determinants (up to 0.68 micrometres).
    budgets = 8 * np.finfo(np.float32).eps * np.max(np.abs(triangles), axis=(1, 2))
    errors = np.linalg.norm(result - expected, axis=1)
    test.assertTrue(np.all(errors <= budgets), f"maximum reconstruction error/budget: {np.max(errors / budgets)}")
    uv = barycentric.numpy()
    test.assertTrue(np.isfinite(uv).all())
    test.assertTrue(np.all(uv >= -2e-7))
    test.assertTrue(np.all(uv.sum(axis=1) <= 1.0 + 2e-7))


def test_mesh_query_point_close_faces(test, device):
    vertices, indices, points = closest_point_precision_fixture()
    expected = points.astype(np.float64)
    expected[:, 1] = float(vertices[0, 1])
    # The correct face is strictly closer for both points, and its projections
    # are interior. Reordering either face must not change that answer.
    for order, permutation in itertools.product(([0, 1], [1, 0]), itertools.permutations(range(3))):
        faces = indices[list(order)][:, list(permutation)]
        with test.subTest(order=order, permutation=permutation):
            mesh = wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3, device=device),
                indices=wp.array(faces.ravel(), dtype=int, device=device),
            )
            closest = wp.empty(len(points), dtype=wp.vec3, device=device)
            found_faces = wp.empty(len(points), dtype=int, device=device)
            wp.launch(
                closest_point_precision_mesh_kernel,
                dim=len(points),
                inputs=[mesh.id, wp.array(points, dtype=wp.vec3, device=device)],
                outputs=[closest, found_faces],
                device=device,
            )
            np.testing.assert_array_equal(found_faces.numpy(), np.full(len(points), order.index(1)))
            result = closest.numpy().astype(np.float64)
            np.testing.assert_allclose(result, expected, rtol=0, atol=8e-9)
            directions = points.astype(np.float64) - result
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            np.testing.assert_allclose(directions, np.tile([0.0, 1.0, 0.0], (len(points), 1)), rtol=0, atol=2e-3)


def test_closest_point_triangle_precision_grad(test, device):
    vertices, indices, points = closest_point_precision_fixture()
    thin = vertices[indices[1]]
    regular = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    queries = np.concatenate(
        (
            points,
            np.array(
                [
                    [0.2, 0.3, 0.4],
                    [0.3, -0.2, 0.4],
                    [-0.2, 0.3, 0.4],
                    [0.7, 0.7, 0.4],
                    [-0.2, -0.1, 0.4],
                    [1.2, -0.1, 0.4],
                    [-0.1, 1.2, 0.4],
                ],
                np.float32,
            ),
        )
    )
    triangles = np.array([thin, thin] + [regular] * 7, dtype=np.float32)
    weights = np.array([0.3, -0.2, 0.7], dtype=np.float32)
    tri_wp = wp.array(triangles.reshape(-1, 3), dtype=wp.vec3, device=device, requires_grad=True)
    point_wp = wp.array(queries, dtype=wp.vec3, device=device, requires_grad=True)
    closest = wp.empty(len(queries), dtype=wp.vec3, device=device, requires_grad=True)
    barycentric = wp.empty(len(queries), dtype=wp.vec2, device=device)
    with wp.Tape() as tape:
        wp.launch(
            closest_point_precision_kernel,
            dim=len(queries),
            inputs=[tri_wp, point_wp],
            outputs=[closest, barycentric],
            device=device,
        )
    tape.backward(grads={closest: wp.array(np.tile(weights, (len(queries), 1)), dtype=wp.vec3, device=device)})
    expected = np.empty((len(queries), 4, 3))
    for i, (triangle, point) in enumerate(zip(triangles, queries, strict=True)):
        # These points stay in one smooth feature region for the entire stencil.
        variables = np.vstack((triangle, point)).astype(np.float64)
        epsilon = 1e-8 if i < 2 else 1e-5
        for row, coordinate in itertools.product(range(4), range(3)):
            plus, minus = variables.copy(), variables.copy()
            plus[row, coordinate] += epsilon
            minus[row, coordinate] -= epsilon
            p = closest_point_triangle_reference(plus[:3], plus[3])
            m = closest_point_triangle_reference(minus[:3], minus[3])
            expected[i, row, coordinate] = np.dot(weights, p - m) / (2 * epsilon)
    actual = np.concatenate((tri_wp.grad.numpy().reshape(-1, 3, 3), point_wp.grad.numpy()[:, None, :]), axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_closest_point_triangle_boundary_grad(test, device):
    triangles, points = closest_point_boundary_fixture()
    # The original inputs are too near a feature boundary for an ordinary
    # finite-difference stencil. Use the analytic finite-edge derivative and
    # also check controls farther outside the same two edges.
    edge_ids = [(0, 2), (1, 2)]
    controls = []
    for triangle, point, (i, j) in zip(triangles, points, edge_ids, strict=True):
        tri = triangle.astype(np.float64)
        p = point.astype(np.float64)
        a, b = tri[i], tri[j]
        edge = b - a
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal /= np.linalg.norm(normal)
        projected = a + np.dot(p - a, edge) / np.dot(edge, edge) * edge
        outward = p - projected - np.dot(p - projected, normal) * normal
        outward /= np.linalg.norm(outward)
        controls.append(p + 1e-4 * np.linalg.norm(edge) * outward)
    triangles = np.concatenate((triangles, triangles))
    points = np.concatenate((points, np.asarray(controls, np.float32)))
    edge_ids *= 2
    weights = np.array([0.3, -0.2, 0.7], dtype=np.float32)
    tri_wp = wp.array(triangles.reshape(-1, 3), dtype=wp.vec3, device=device, requires_grad=True)
    point_wp = wp.array(points, dtype=wp.vec3, device=device, requires_grad=True)
    closest = wp.empty(len(points), dtype=wp.vec3, device=device, requires_grad=True)
    barycentric = wp.empty(len(points), dtype=wp.vec2, device=device)
    with wp.Tape() as tape:
        wp.launch(
            closest_point_precision_kernel,
            dim=len(points),
            inputs=[tri_wp, point_wp],
            outputs=[closest, barycentric],
            device=device,
        )
    tape.backward(grads={closest: wp.array(np.tile(weights, (len(points), 1)), dtype=wp.vec3, device=device)})
    expected = np.zeros((len(points), 4, 3))
    reference_weights = weights.astype(np.float64)
    for row, (triangle, point, (i, j)) in enumerate(zip(triangles, points, edge_ids, strict=True)):
        a, b = triangle.astype(np.float64)[[i, j]]
        edge = b - a
        offset = point.astype(np.float64) - a
        length_sq = np.dot(edge, edge)
        t = np.dot(offset, edge) / length_sq
        test.assertGreater(t, 0.0)
        test.assertLess(t, 1.0)
        along = np.dot(reference_weights, edge)
        expected[row, i] = (1.0 - t) * reference_weights + along * ((2.0 * t - 1.0) * edge - offset) / length_sq
        expected[row, j] = t * reference_weights + along * (offset - 2.0 * t * edge) / length_sq
        expected[row, 3] = along * edge / length_sq
        # The opposite vertex has no derivative in this smooth edge region.
        np.testing.assert_allclose(expected[row].sum(axis=0), weights, rtol=0, atol=1e-12)
    actual = np.concatenate((tri_wp.grad.numpy().reshape(-1, 3, 3), point_wp.grad.numpy()[:, None, :]), axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def mesh_query_relative_distance_fixture():
    vertices, faces, _ = closest_point_precision_fixture()
    # Eight distinct native float32 inputs that still selected the bevel after
    # fixing closest_point_to_triangle's Gram-determinant cancellation. All
    # have a strictly closer interior projection on the neighboring flat face.
    points = np.array(
        [
            [1017209282, 1011562211, 3142393395],
            [1017209297, 1011562223, 3142393550],
            [1017209293, 1011562212, 3142393547],
            [1017209297, 1011562213, 3142393480],
            [1017209301, 1011562213, 3142393693],
            [1017209305, 1011562212, 3142393606],
            [1017209313, 1011562210, 3142393678],
            [1017209313, 1011562207, 3142393660],
        ],
        dtype=np.uint32,
    ).view(np.float32)
    return vertices, faces, points


@wp.func
def mesh_query_relative_mode(mesh: wp.uint64, point: wp.vec3, radius: float, mode: int):
    if mode == 0:
        return wp.mesh_query_point(mesh, point, radius)
    elif mode == 1:
        return wp.mesh_query_point_sign_parity(mesh, point, radius)
    elif mode == 2:
        return wp.mesh_query_point_sign_normal(mesh, point, radius)
    elif mode == 3:
        return wp.mesh_query_point_sign_winding_number(mesh, point, radius)
    else:
        return wp.mesh_query_point_no_sign(mesh, point, radius)


@wp.kernel
def mesh_query_relative_kernel(
    mesh: wp.uint64,
    points: wp.array[wp.vec3],
    radii: wp.array[float],
    results: wp.array2d[float],
    closest: wp.array[wp.vec3],
):
    i = wp.tid()
    query = mesh_query_relative_mode(mesh, points[i], radii[i], i % 5)
    results[i, 0] = float(query.result)
    if query.result:
        results[i, 1] = float(query.face)
        results[i, 2] = query.sign
        results[i, 3] = query.u
        results[i, 4] = query.v
        closest[i] = wp.mesh_eval_position(mesh, query.face, query.u, query.v)


def mesh_query_relative_results(mesh, points, radii):
    points = np.repeat(np.asarray(points, np.float32), 5, axis=0)
    radii = np.repeat(np.broadcast_to(np.asarray(radii, np.float32), (len(points) // 5,)), 5)
    results = wp.zeros((len(points), 5), dtype=float, device=mesh.device)
    closest = wp.zeros(len(points), dtype=wp.vec3, device=mesh.device)
    wp.launch(
        mesh_query_relative_kernel,
        dim=len(points),
        inputs=[
            mesh.id,
            wp.array(points, dtype=wp.vec3, device=mesh.device),
            wp.array(radii, dtype=float, device=mesh.device),
        ],
        outputs=[results, closest],
        device=mesh.device,
    )
    return results.numpy().reshape(-1, 5, 5), closest.numpy().reshape(-1, 5, 3)


def test_mesh_query_relative_distance(test, device):
    vertices, faces, points = mesh_query_relative_distance_fixture()
    expected = points.copy()
    expected[:, 1] = vertices[0, 1]
    # Confirm strict finite-face uniqueness independently, rather than requiring
    # a particular face for genuinely tied queries.
    for point, reference in zip(points, expected, strict=True):
        flat = closest_point_triangle_reference(vertices[faces[1]], point)
        bevel = closest_point_triangle_reference(vertices[faces[0]], point)
        np.testing.assert_allclose(flat, reference, rtol=0, atol=1e-16)
        test.assertLess(np.linalg.norm(flat - point), np.linalg.norm(bevel - point))
    for order, permutation, axes, expanded in itertools.product(
        ((0, 1), (1, 0)), itertools.permutations(range(3)), ((0, 1, 2), (2, 0, 1)), (False, True)
    ):
        with test.subTest(order=order, permutation=permutation, axes=axes, expanded=expanded):
            v = vertices[:, axes].copy()
            f = faces[list(order)][:, permutation].copy()
            if expanded:
                # Force traversal beyond the two-face leaf without changing the
                # nearest surface. These distant triangles are all nondegenerate.
                for i in range(8):
                    f = np.vstack((f, np.arange(len(v), len(v) + 3)))
                    v = np.vstack((v, vertices[faces[0]][:, axes] + np.array([0.1 + 0.01 * i, 0.1, 0.1])))
            mesh = wp.Mesh(
                points=wp.array(v, dtype=wp.vec3, device=device),
                indices=wp.array(f.ravel(), dtype=int, device=device),
                support_winding_number=True,
                bvh_leaf_size=1,
            )
            result, closest = mesh_query_relative_results(mesh, points[:, axes], 4e-6)
            np.testing.assert_array_equal(result[:, :, 0], 1.0)
            np.testing.assert_array_equal(result[:, :, 1], order.index(1))
            # mesh_eval_position remains an absolute-coordinate reconstruction;
            # its output accuracy is distinct from the internal distance ranking.
            np.testing.assert_allclose(closest, np.repeat(expected[:, None, axes], 5, axis=1), rtol=0, atol=8e-9)
            weights = np.concatenate((result[:, :, 3:5], (1.0 - result[:, :, 3:4] - result[:, :, 4:5])), axis=2)
            test.assertGreaterEqual(float(weights.min()), -2 * np.finfo(np.float32).eps)
            # The fixture is open and vertex permutations may reverse winding;
            # no solid-containment claim is made from its sign results.


def test_mesh_query_relative_radius_and_sign(test, device):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.ravel(), dtype=int, device=device),
        support_winding_number=True,
        bvh_leaf_size=1,
    )
    points = np.array([[0.25, 0.25, -0.125], [0.25, 0.25, 0.125]], np.float32)
    radius = np.float32(0.125)
    for r, found in (
        (np.nextafter(radius, np.float32(0)), False),
        (radius, False),
        (np.nextafter(radius, np.float32(np.inf)), True),
    ):
        result, closest = mesh_query_relative_results(mesh, points, r)
        np.testing.assert_array_equal(result[:, :, 0], float(found))
        if found:
            np.testing.assert_array_equal(result[:, :, 1], 0)
            np.testing.assert_array_equal(result[:, :4, 2], np.tile([[1.0], [-1.0]], (1, 4)))
            np.testing.assert_array_equal(closest, np.tile([0.25, 0.25, 0.0], (2, 5, 1)))
    # Exact vertex projections exercise finite-segment endpoints t=0 and t=1.
    # Cyclic permutations preserve the closed surface's outward winding. Any
    # incident tied face is valid; only the geometric result and sign are fixed.
    points = np.array([[0, 0, -0.25], [1, 0, -0.25], [0, 1, -0.25]], np.float32)
    expected = vertices[:3]
    incident = ({0, 1, 2}, {0, 1, 3}, {0, 2, 3})
    radius = np.float32(0.25)
    for permutation in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=device),
            indices=wp.array(faces[:, permutation].ravel(), dtype=int, device=device),
            support_winding_number=True,
            bvh_leaf_size=1,
        )
        for r, found in (
            (np.nextafter(radius, np.float32(0)), False),
            (radius, False),
            (np.nextafter(radius, np.float32(np.inf)), True),
        ):
            result, closest = mesh_query_relative_results(mesh, points, r)
            np.testing.assert_array_equal(result[:, :, 0], float(found))
            if found:
                for row, valid_faces in enumerate(incident):
                    test.assertTrue(set(result[row, :, 1]).issubset(valid_faces))
                np.testing.assert_array_equal(result[:, :4, 2], 1.0)
                np.testing.assert_array_equal(closest, np.repeat(expected[:, None, :], 5, axis=1))
    # Tight, finite search radii around the historical queries have explicit
    # geometric headroom; do not conflate their nonbinary distances with the
    # exact one-ULP boundary control above.
    vertices, faces, points = mesh_query_relative_distance_fixture()
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.ravel(), dtype=int, device=device),
        support_winding_number=True,
    )
    distance = points[:, 1] - vertices[0, 1]
    for offset, found in ((-4e-9, False), (4e-9, True)):
        result, _ = mesh_query_relative_results(mesh, points, distance + offset)
        np.testing.assert_array_equal(result[:, :, 0], float(found))
        if found:
            np.testing.assert_array_equal(result[:, :, 1], 1)


def test_mesh_query_relative_tie(test, device):
    # The exact orthogonal projection is on a shared edge. Either face is a
    # valid answer; face order must not be mistaken for a geometric failure.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    for order in ((0, 1), (1, 0)):
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=device),
            indices=wp.array(faces[list(order)].ravel(), dtype=int, device=device),
            support_winding_number=True,
        )
        result, closest = mesh_query_relative_results(mesh, [[0.5, 0.5, 0.125]], 0.25)
        np.testing.assert_array_equal(result[:, :, 0], 1.0)
        test.assertTrue(np.isin(result[:, :, 1], [0, 1]).all())
        np.testing.assert_array_equal(closest, np.tile([0.5, 0.5, 0], (1, 5, 1)))


def test_mesh_query_relative_grad(test, device):
    vertices, faces, points = mesh_query_relative_distance_fixture()
    mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.ravel(), dtype=int, device=device),
        support_winding_number=True,
    )
    points = np.repeat(points, 5, axis=0)
    point_wp = wp.array(points, dtype=wp.vec3, device=device, requires_grad=True)
    radii = wp.full(len(points), 4e-6, dtype=float, device=device, requires_grad=True)
    results = wp.zeros((len(points), 5), dtype=float, device=device)
    closest = wp.zeros(len(points), dtype=wp.vec3, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(
            mesh_query_relative_kernel,
            dim=len(points),
            inputs=[mesh.id, point_wp, radii],
            outputs=[results, closest],
            device=device,
        )
    tape.backward(grads={closest: wp.array(np.tile([0.3, -0.2, 0.7], (len(points), 1)), dtype=wp.vec3, device=device)})
    np.testing.assert_array_equal(results.numpy()[:, 1], 1)
    # The selected flat is smooth at each point. Its projection derivative is
    # diag(1,0,1); face selection and the query radius are not differentiated.
    np.testing.assert_allclose(point_wp.grad.numpy(), np.tile([0.3, 0.0, 0.7], (len(points), 1)), rtol=2e-4, atol=2e-4)
    np.testing.assert_array_equal(radii.grad.numpy(), 0.0)


devices = get_test_devices()


class TestMeshQueryPoint(unittest.TestCase):
    def test_mesh_query_point_codegen_adjoints_with_select(self):
        def kernel_fn(
            mesh: wp.uint64,
        ):
            v = wp.vec3(0.0, 0.0, 0.0)
            d = 1e-6

            if True:
                query_1 = wp.mesh_query_point(mesh, v, d)
                query_2 = wp.mesh_query_point_no_sign(mesh, v, d)
                query_3 = wp.mesh_query_furthest_point_no_sign(mesh, v, d)
                query_4 = wp.mesh_query_point_sign_normal(mesh, v, d)
                query_5 = wp.mesh_query_point_sign_winding_number(mesh, v, d)
            else:
                query_1 = wp.mesh_query_point(mesh, v, d)
                query_2 = wp.mesh_query_point_no_sign(mesh, v, d)
                query_3 = wp.mesh_query_furthest_point_no_sign(mesh, v, d)
                query_4 = wp.mesh_query_point_sign_normal(mesh, v, d)
                query_5 = wp.mesh_query_point_sign_winding_number(mesh, v, d)

        wp.Kernel(func=kernel_fn)


add_function_test(TestMeshQueryPoint, "test_mesh_query_point", test_mesh_query_point, devices=devices)
add_function_test(TestMeshQueryPoint, "test_mesh_query_furthest_point", test_mesh_query_furthest_point, devices=devices)
add_function_test(TestMeshQueryPoint, "test_adj_mesh_query_point", test_adj_mesh_query_point, devices=devices)
add_function_test(TestMeshQueryPoint, "test_set_mesh_points", test_set_mesh_points, devices=devices)
add_function_test(
    TestMeshQueryPoint, "test_mesh_query_parity_outside_bounds", test_mesh_query_parity_outside_bounds, devices=devices
)
add_function_test(
    TestMeshQueryPoint,
    "test_mesh_query_parity_bounds_sample_count",
    test_mesh_query_parity_bounds_sample_count,
    devices=devices,
)
add_function_test(
    TestMeshQueryPoint, "test_mesh_query_parity_bounds_refit", test_mesh_query_parity_bounds_refit, devices=devices
)

add_function_test(
    TestMeshQueryPoint, "test_closest_point_triangle_precision", test_closest_point_triangle_precision, devices=devices
)
add_function_test(
    TestMeshQueryPoint, "test_mesh_query_point_close_faces", test_mesh_query_point_close_faces, devices=devices
)
add_function_test(
    TestMeshQueryPoint,
    "test_closest_point_triangle_precision_grad",
    test_closest_point_triangle_precision_grad,
    devices=devices,
)

add_function_test(
    TestMeshQueryPoint,
    "test_closest_point_triangle_boundary_grad",
    test_closest_point_triangle_boundary_grad,
    devices=devices,
)

add_function_test(
    TestMeshQueryPoint, "test_mesh_query_relative_distance", test_mesh_query_relative_distance, devices=devices
)
add_function_test(
    TestMeshQueryPoint,
    "test_mesh_query_relative_radius_and_sign",
    test_mesh_query_relative_radius_and_sign,
    devices=devices,
)
add_function_test(TestMeshQueryPoint, "test_mesh_query_relative_tie", test_mesh_query_relative_tie, devices=devices)
add_function_test(TestMeshQueryPoint, "test_mesh_query_relative_grad", test_mesh_query_relative_grad, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
