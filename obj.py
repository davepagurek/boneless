def parse_obj(path):
    vertices = []
    edges = []
    faces = []

    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = [float(loc) for loc in line[2:].split(" ")]
                vertices.append((x, y, z))
            elif line.startswith("f "):
                indices = [int(idx)-1 for idx in line[2:].split(" ")]
                faces.append(indices)
                for i in range(len(indices)):
                    idx_a = indices[i]
                    idx_b = indices[(i+1) % len(indices)]
                    edges.append((idx_a, idx_b))

    return vertices, edges, faces
