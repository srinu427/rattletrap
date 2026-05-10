struct Vec4 {
    x @0: Float32,
    y @1: Float32,
    z @2: Float32,
    w @3: Float32,
}

struct Vertex {
    pos @0: Vec4,
    uv @1: Vec4,
    n @2: Vec4,
    t @3: Vec4,
    bt @4: Vec4,
}

struct Mesh {
    verts @0: List(Vertex),
    indxs @1: List(UInt16),
}