#!/usr/bin/env python3
"""Turn a foot GLB into the compact mesh the FOOT MAP loads (assets/foot.bin).

    python3 tools/prep_foot.py MODEL.glb [--node NAME] [--out assets/foot.bin]

What it does
  1. Picks one node subtree (default: the one with the most vertices) and merges
     all its mesh primitives into a single mesh, in world space.
  2. Puts it in the app's frame: Y up with the sole on y=0, foot length along Z
     scaled to 26 units (heel -13, toes +13), centred on X. Only a proper rotation
     is used, so a right foot stays a right foot.
  3. Works out which X side is medial (big-toe side) from three independent cues
     and records it, so the app can label medial/lateral and migrate old strokes.
  4. Writes positions (int16, /1000), normals (int8, /127) and uint16 indices.

Binary layout (little-endian)
  0  'SNKF'            4  u16 version=1, u16 flags=0
  8  u32 vertCount     12 u32 indexCount
  16 i32 medialSign    20 f32 posScale (=0.001)
  24 f32 length        28 f32 height
  32 i16[3V] positions, pad4, i8[3V] normals, pad4, u16[indexCount] indices
"""
import argparse, json, math, os, struct, sys

LENGTH = 26.0


def load_glb(path):
    b = open(path, 'rb').read()
    if b[:4] != b'glTF':
        sys.exit('not a GLB file')
    off, J, BIN = 12, None, None
    while off < len(b):
        l, t = struct.unpack_from('<I4s', b, off)
        if t == b'JSON':
            J = json.loads(b[off + 8:off + 8 + l])
        elif t == b'BIN\x00':
            BIN = b[off + 8:off + 8 + l]
        off += 8 + l
    if (J.get('extensionsRequired') or []):
        sys.exit('compressed GLB (%s) not supported; export uncompressed' % J['extensionsRequired'])
    return J, BIN


CT = {5126: ('f', 4), 5123: ('H', 2), 5125: ('I', 4), 5121: ('B', 1)}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def read_acc(J, BIN, ai):
    a = J['accessors'][ai]; v = J['bufferViews'][a['bufferView']]
    fmt, sz = CT[a['componentType']]; nc = NC[a['type']]
    base = v.get('byteOffset', 0) + a.get('byteOffset', 0); stride = v.get('byteStride', sz * nc)
    return [struct.unpack_from('<' + fmt * nc, BIN, base + i * stride) for i in range(a['count'])]


def node_matrix(n):
    if 'matrix' in n:
        m = n['matrix']; return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    t = n.get('translation', [0, 0, 0]); q = n.get('rotation', [0, 0, 0, 1]); s = n.get('scale', [1, 1, 1])
    x, y, z, w = q
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    return [[R[r][0] * s[0], R[r][1] * s[1], R[r][2] * s[2], t[r]] for r in range(3)] + [[0, 0, 0, 1]]


def mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def normal_matrix(M):
    a = [[M[r][c] for c in range(3)] for r in range(3)]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    inv = [[(a[(j + 1) % 3][(i + 1) % 3] * a[(j + 2) % 3][(i + 2) % 3]
             - a[(j + 1) % 3][(i + 2) % 3] * a[(j + 2) % 3][(i + 1) % 3]) / det for j in range(3)] for i in range(3)]
    return [[inv[c][r] for c in range(3)] for r in range(3)], det


def subtree_meshes(J, root):
    out, stack = [], [root]
    while stack:
        i = stack.pop(); n = J['nodes'][i]
        if 'mesh' in n: out.append(i)
        stack.extend(n.get('children', []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('glb'); ap.add_argument('--node'); ap.add_argument('--out', default='assets/foot.bin')
    args = ap.parse_args()
    J, BIN = load_glb(args.glb)
    nodes = J['nodes']; parent = {}
    for i, n in enumerate(nodes):
        for c in n.get('children', []): parent[c] = i

    def world(i):
        M = node_matrix(nodes[i])
        while i in parent:
            i = parent[i]; M = mul(node_matrix(nodes[i]), M)
        return M

    def vcount(i):
        return sum(J['accessors'][p['attributes']['POSITION']]['count']
                   for m in subtree_meshes(J, i) for p in J['meshes'][nodes[m]['mesh']]['primitives'])

    if args.node:
        roots = [i for i, n in enumerate(nodes) if n.get('name') == args.node]
        if not roots: sys.exit('node %r not found' % args.node)
        root = roots[0]
    else:  # biggest subtree that is not an ancestor-of-everything
        cands = [i for i, n in enumerate(nodes) if n.get('children') and any('mesh' in nodes[c] for c in n['children'])]
        root = max(cands or range(len(nodes)), key=vcount)
    print('node      : %s (%d verts)' % (nodes[root].get('name'), vcount(root)))

    P, N, I = [], [], []
    for mi in subtree_meshes(J, root):
        W = world(mi); NM, _ = normal_matrix(W)
        for pr in J['meshes'][nodes[mi]['mesh']]['primitives']:
            if pr.get('mode', 4) != 4: continue
            pos = read_acc(J, BIN, pr['attributes']['POSITION'])
            nor = read_acc(J, BIN, pr['attributes']['NORMAL']) if 'NORMAL' in pr['attributes'] else None
            idx = [x[0] for x in read_acc(J, BIN, pr['indices'])] if 'indices' in pr else list(range(len(pos)))
            o = len(P)
            for (x, y, z) in pos:
                P.append([W[r][0] * x + W[r][1] * y + W[r][2] * z + W[r][3] for r in range(3)])
            for k in range(len(pos)):
                if nor:
                    x, y, z = nor[k]; v = [NM[r][0] * x + NM[r][1] * y + NM[r][2] * z for r in range(3)]
                else:
                    v = [0, 1, 0]
                L = math.sqrt(sum(q * q for q in v)) or 1; N.append([q / L for q in v])
            I += [k + o for k in idx]
    if len(P) >= 65536: sys.exit('too many vertices for uint16 indices (%d); decimate first' % len(P))

    # ---- orientation: glTF is Y-up; length is the longer horizontal extent ----
    ext = lambda a: (min(p[a] for p in P), max(p[a] for p in P))
    (x0, x1), (y0, y1), (z0, z1) = ext(0), ext(1), ext(2)
    la = 0 if (x1 - x0) >= (z1 - z0) else 2
    lo, hi = (x0, x1) if la == 0 else (z0, z1); Lw = hi - lo
    top = lambda a, b: max((p[1] for p in P if a <= p[la] <= b), default=y0)
    low_end_h, high_end_h = top(lo, lo + 0.25 * Lw), top(hi - 0.25 * Lw, hi)
    toe_dir = -1 if low_end_h < high_end_h else 1        # toes sit at the lower end (ankle is at the heel)
    zrow = [0, 0, 0]; zrow[la] = float(toe_dir)          # our +Z (toes) in world basis
    yrow = [0, 1, 0]
    xrow = [yrow[1] * zrow[2] - yrow[2] * zrow[1], yrow[2] * zrow[0] - yrow[0] * zrow[2], yrow[0] * zrow[1] - yrow[1] * zrow[0]]
    s = LENGTH / Lw
    c = [(x0 + x1) / 2, y0, (z0 + z1) / 2]
    dot = lambda r, v: r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
    Q = [[dot(xrow, [p[i] - c[i] for i in range(3)]) * s, (p[1] - y0) * s, dot(zrow, [p[i] - c[i] for i in range(3)]) * s] for p in P]
    NN = [[dot(xrow, n), n[1], dot(zrow, n)] for n in N]
    xc = (min(q[0] for q in Q) + max(q[0] for q in Q)) / 2
    for q in Q: q[0] -= xc
    zmid = (min(q[2] for q in Q) + max(q[2] for q in Q)) / 2
    for q in Q: q[2] -= zmid

    # ---- medial side: three independent cues, majority vote ----
    t = lambda q: (q[2] + LENGTH / 2) / LENGTH
    toes = [q for q in Q if t(q) > 0.88]; mid = [q for q in Q if 0.35 < t(q) < 0.60]
    side = lambda cond: 1 if cond else -1
    cue_toe = side(max((q[1] for q in toes if q[0] > 0), default=0) > max((q[1] for q in toes if q[0] <= 0), default=0))
    cue_arch = side(min((q[1] for q in mid if q[0] > 0), default=0) > min((q[1] for q in mid if q[0] <= 0), default=0))
    cue_tip = side(max(Q, key=t)[0] > 0)
    votes = cue_toe + cue_arch + cue_tip
    medial = 1 if votes > 0 else -1
    print('medial    : %+d  (thicker toe %+d, raised arch %+d, foremost tip %+d)%s' % (
        medial, cue_toe, cue_arch, cue_tip, '' if abs(votes) == 3 else '  <- cues disagree, check visually'))
    print('foot      : %s foot' % ('right' if medial > 0 else 'left'))
    height = max(q[1] for q in Q); width = max(q[0] for q in Q) - min(q[0] for q in Q)
    print('frame     : length %.1f  height %.1f  width %.1f  (units ~ cm)' % (LENGTH, height, width))
    print('mesh      : %d verts, %d tris' % (len(Q), len(I) // 3))

    # ---- write ----
    V, IC = len(Q), len(I)
    buf = bytearray(b'SNKF') + struct.pack('<HHIIiff f'.replace(' ', ''), 1, 0, V, IC, medial, 0.001, LENGTH, height)
    assert len(buf) == 32
    buf += struct.pack('<%dh' % (3 * V), *[max(-32767, min(32767, round(v * 1000))) for q in Q for v in q])
    buf += b'\0' * (-len(buf) % 4)
    buf += struct.pack('<%db' % (3 * V), *[max(-127, min(127, round(v * 127))) for n in NN for v in n])
    buf += b'\0' * (-len(buf) % 4)
    buf += struct.pack('<%dH' % IC, *I)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, 'wb').write(buf)
    print('wrote     : %s (%.0f KB)' % (args.out, len(buf) / 1024))
    extras = (J.get('asset') or {}).get('extras') or {}
    if extras:
        print('credit    : "%s" by %s — %s — %s' % (extras.get('title'), extras.get('author'), extras.get('license'), extras.get('source')))


if __name__ == '__main__':
    main()
