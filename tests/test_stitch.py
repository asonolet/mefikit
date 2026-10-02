import json

import numpy as np
import pytest

import mefikit as mf

_SEP = np.iinfo(np.uint64).max


def _blocks(mesh):
    """Yields `(element_type, cell connectivity)` for every cell."""
    for et, conn in mesh.blocks().items():
        if isinstance(conn, tuple):
            data, offsets = conn
            prev = 0
            for off in offsets:
                yield et, np.asarray(data[prev:off])
                prev = off
        else:
            for cell in np.asarray(conn):
                yield et, cell


def _faces(cell):
    """Splits a PHED connectivity into its faces."""
    faces, cur = [], []
    for n in cell:
        if n == _SEP:
            faces.append([int(g) for g in cur])
            cur = []
        else:
            cur.append(int(n))
    if cur:
        faces.append([int(g) for g in cur])
    return faces


def _families(mesh):
    block = json.loads(mesh.to_json())["element_blocks"]
    return {et: np.asarray(info["families"]["data"]) for et, info in block.items()}


def _faces_on_plane(mesh, axis, value, tol=1e-12):
    """Maps each face lying on `coords[axis] == value` to how many cells use it."""
    coords = np.asarray(mesh.coords())
    counts = {}
    for _, conn in _blocks(mesh):
        for face in _faces(conn):
            if all(abs(coords[g, axis] - value) <= tol for g in face):
                key = tuple(sorted(face))
                counts[key] = counts.get(key, 0) + 1
    return counts


def _box(*axes):
    return mf.build_cmesh(*axes)


def _rotated_box(angle, centre, *axes):
    """Rotates a box by `angle` radians about the axis through `centre` parallel to `z`."""
    return mf.build_cmesh(*axes).rotate_about(centre, [0.0, 0.0, 1.0], angle)


def _interface_area(mesh):
    """Total area of the faces shared by two cells of different families."""
    coords = np.asarray(mesh.coords())
    families = _families(mesh)["PHED"]
    faces = {}
    for i, (_, conn) in enumerate(_blocks(mesh)):
        for face in _faces(conn):
            faces.setdefault(tuple(sorted(face)), (face, []))[1].append(i)
    total = 0.0
    for ring, users in faces.values():
        if len(users) != 2 or families[users[0]] == families[users[1]]:
            continue
        pts = coords[ring][:, :2]
        x, y = pts[:, 0], pts[:, 1]
        total += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return total


def _face_sizes(mesh):
    """Returns the number of nodes of every distinct face of the mesh."""
    keys = {tuple(sorted(face)) for _, conn in _blocks(mesh) for face in _faces(conn)}
    return sorted(len(key) for key in keys)


def test_stitch_requires_two_meshes():
    with pytest.raises(ValueError, match="at least two meshes"):
        mf.stitch([_box([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])])


def test_stitch_rejects_bad_tolerance():
    a = _box([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    b = _box([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="tolerance"):
        mf.stitch([a, b], tol=-1.0)


def test_stitch_rejects_surface_mesh():
    a = _box([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    b = _box([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="3d space"):
        mf.stitch([a, b])


def test_stitch_matching_interfaces():
    a = _box([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    b = _box([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])
    out = mf.stitch([a, b])

    assert len(list(_blocks(out))) == 2
    assert {et for et, _ in _blocks(out)} == {"PHED"}
    assert out.coords().shape == (12, 3)
    assert list(_faces_on_plane(out, 2, 1.0).values()) == [2]


def test_stitch_mismatched_interfaces():
    # One big hex against four hexes: the shared interface is 1 quad vs 4 quads.
    a = _box([0.0, 2.0], [0.0, 2.0], [0.0, 1.0])
    b = _box([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 2.0])
    out = mf.stitch([a, b])

    assert len(list(_blocks(out))) == 5
    # 8 + 18 nodes, 4 of which are the shared interface corners.
    assert out.coords().shape == (22, 3)
    # The single quad of A became 4 quads, each shared with one of B's quads.
    assert len(_faces_on_plane(out, 2, 1.0)) == 4
    assert set(_faces_on_plane(out, 2, 1.0).values()) == {2}


def test_stitch_partial_overlap():
    # B's bottom face sticks out of A's top face on the +x side.
    a = _box([0.0, 2.0], [0.0, 2.0], [0.0, 1.0])
    b = _box([0.0, 3.0], [0.0, 1.0], [1.0, 2.0])
    out = mf.stitch([a, b])

    counts = _faces_on_plane(out, 2, 1.0)
    assert len(counts) == 3
    assert sorted(counts.values()) == [1, 1, 2]


def test_stitch_three_meshes():
    a = _box([0.0, 2.0], [0.0, 2.0], [0.0, 1.0])
    b = _box([0.0, 1.0, 2.0], [0.0, 2.0], [1.0, 2.0])
    c = _box([0.0, 2.0], [0.0, 1.0, 2.0], [2.0, 3.0])
    out = mf.stitch([a, b, c])

    assert len(list(_blocks(out))) == 5
    assert set(_faces_on_plane(out, 2, 1.0).values()) == {2}
    assert len(_faces_on_plane(out, 2, 1.0)) == 2
    assert set(_faces_on_plane(out, 2, 2.0).values()) == {2}
    assert len(_faces_on_plane(out, 2, 2.0)) == 4


def test_stitch_shares_interface_nodes():
    a = _box([0.0, 2.0], [0.0, 2.0], [0.0, 1.0])
    b = _box([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 2.0])
    out = mf.stitch([a, b])
    coords = np.asarray(out.coords())

    # The first cell comes from `a`, the remaining four from `b`. The interface nodes of `a` are
    # exactly those of `b`, and both are the 9 nodes of the interface plane.
    cells = list(_blocks(out))
    assert len(cells) == 5
    a_nodes = {g for face in _faces(cells[0][1]) for g in face}
    b_nodes = {g for _, conn in cells[1:] for face in _faces(conn) for g in face}
    a_iface = {g for g in a_nodes if abs(coords[g, 2] - 1.0) < 1e-12}
    b_iface = {g for g in b_nodes if abs(coords[g, 2] - 1.0) < 1e-12}
    assert a_iface == b_iface
    assert len(a_iface) == 9


def test_stitch_families_are_relabeled():
    a = _box([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    b = _box([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])
    out = mf.stitch([a, b])

    families = _families(out)
    assert set(families) == {"PHED"}
    assert sorted(set(families["PHED"].tolist())) == [0, 1]
    # The single cell of the first mesh keeps family 0, the second one is shifted to 1.
    assert families["PHED"].tolist() == [0, 1]


def test_stitch_disjoint_meshes():
    a = _box([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    b = _box([5.0, 6.0], [0.0, 1.0], [0.0, 1.0])
    out = mf.stitch([a, b])

    assert len(list(_blocks(out))) == 2
    assert set(_faces_on_plane(out, 2, 1.0).values()) == {1}


def test_stitch_drops_fields_and_groups():
    a = _box([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    b = _box([0.0, 1.0], [0.0, 1.0], [1.0, 2.0])
    a.set_field("phi", {"HEX8": np.zeros((1, 1))})
    b.set_field("phi", {"HEX8": np.zeros((1, 1))})
    b.groups["cells"] = {"HEX8": [0]}

    out = mf.stitch([a, b])
    assert dict(out.fields) == {}
    assert dict(out.groups) == {}


# ------------------------------------------------------------------------------------------
# Hex-dominant cases, including already-conformized (`PHED`) input.
# ------------------------------------------------------------------------------------------


def _l_shaped_hex_block():
    """Three hexes in an L, built by hand since `build_cmesh` only makes full grids."""
    # Node `i + 3j + 9k` sits at `(i, j, k)`.
    coords = np.array(
        [
            [x, y, z]
            for z in (0.0, 1.0)
            for y in (0.0, 1.0, 2.0)
            for x in (0.0, 1.0, 2.0)
        ]
    )
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    def hexa(i, j):
        c = corners + np.array([i, j, 0])
        return c[:, 0] + 3 * c[:, 1] + 9 * c[:, 2]

    conn = np.array([hexa(0, 0), hexa(1, 0), hexa(0, 1)], dtype=np.uint)
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("HEX8", conn)
    return mesh


def test_stitch_accepts_polyzed_hex_input():
    # `polyze` turns an `HEX8` mesh into an already-conformized `PHED` one, which is the kind of
    # block `stitch` is meant for.
    a = _box([0.0, 2.0], [0.0, 2.0], [0.0, 1.0]).polyze()
    b = _box([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 2.0]).polyze()
    assert {et for et, _ in _blocks(a)} == {"PHED"}
    assert {et for et, _ in _blocks(b)} == {"PHED"}

    out = mf.stitch([a, b])

    assert len(list(_blocks(out))) == 5
    assert {et for et, _ in _blocks(out)} == {"PHED"}
    # Every face of the result is a quad: the big face of A was split into the four quads of B.
    assert set(_face_sizes(out)) == {4}


def test_stitch_rotated_grid_conforms_the_interface():
    # The contact region is an octagon, and the rotated grid's own in-plane lines cut it into four
    # pentagons, so both sides have to come back subdivided.
    a = _box([0.0, 2.0], [0.0, 2.0], [0.0, 1.0])
    b = _rotated_box(
        np.pi / 4,
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
        [1.0, 2.0],
    )
    out = mf.stitch([a, b])

    # The octagon is 4 - 2r^2 with r = 2 - sqrt(2), the four corner triangles cut off by the
    # rotated grid. The interface has to cover exactly that.
    r = 2.0 - np.sqrt(2.0)
    assert _interface_area(out) == pytest.approx(4.0 - 2.0 * r * r)

    # Four interface faces, each a pentagon shared by one cell of each mesh. The four quad and
    # four triangle faces that also sit on z = 1 are the corners of A and B that do not overlap.
    counts = _faces_on_plane(out, 2, 1.0)
    assert sorted(counts.values()) == [1] * 8 + [2] * 4
    assert _face_sizes(out).count(5) == 4

    # No two output nodes sit on top of each other, so welding really merged the corners that both
    # meshes bring in.
    coords = np.asarray(out.coords())
    assert len(np.unique(coords, axis=0)) == coords.shape[0]


def test_stitch_hex_block_on_l_shaped_footprint():
    # A concave, three-hex block against a grid whose in-plane lines cut its exposed faces.
    a = _l_shaped_hex_block()
    b = _box([0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0], [0.0, 2.0], [1.0, 2.0])
    assert a.num_elements() == 3
    out = mf.stitch([a, b])

    # The L covers three unit squares, each cut in two by the x line of B that crosses it, and B
    # keeps two faces on z = 1 for the part of its footprint that the L leaves uncovered.
    counts = _faces_on_plane(out, 2, 1.0)
    assert sorted(counts.values()) == [1] * 2 + [2] * 6
    assert _interface_area(out) == pytest.approx(3.0)
    assert _families(out)["PHED"].tolist() == [0, 0, 0, 1, 1, 1]


def test_stitch_offsets_a_hex_block_in_plane():
    # Mismatched in-plane lines: A has one line at x = 1, B has them at 0.5 and 1.5.
    a = _box([0.0, 1.0, 2.0], [0.0, 2.0], [0.0, 1.0])
    b = _box([0.0, 0.5, 1.5, 2.0], [0.0, 2.0], [1.0, 2.0])
    out = mf.stitch([a, b])

    # Each of A's two top faces is cut once by a line of B, so four quads make up the interface.
    counts = _faces_on_plane(out, 2, 1.0)
    assert len(counts) == 4
    assert set(counts.values()) == {2}
    assert _interface_area(out) == pytest.approx(4.0)
    assert _families(out)["PHED"].tolist() == [0, 0, 1, 1, 1]
