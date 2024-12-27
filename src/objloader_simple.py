import os
import cv2


class MTL:

    def __init__(self, filename):
        """Parses a Wavefront MTL file."""
        self.materials = {}
        current_material = None
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Material file '{filename}' not found.")

        base_path = os.path.dirname(filename)

        with open(filename, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                values = line.split()
                if not values:
                    continue
                if values[0] == "newmtl":
                    current_material = values[1]
                    self.materials[current_material] = {}
                elif current_material:
                    if values[0] == "Kd":  # Diffuse color
                        self.materials[current_material]["Kd"] = list(
                            map(float, values[1:4])
                        )
                    elif values[0] == "Ka":  # Ambient color
                        self.materials[current_material]["Ka"] = list(
                            map(float, values[1:4])
                        )
                    elif values[0] == "Ks":  # Specular color
                        self.materials[current_material]["Ks"] = list(
                            map(float, values[1:4])
                        )
                    elif values[0] == "d":  # Transparency
                        self.materials[current_material]["d"] = float(values[1])
                    elif values[0] == "map_Kd":  # Diffuse texture map
                        texture_path = os.path.join(base_path, values[1])
                        texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
                        resized_texture = self.preprocess_texture(texture)
                        self.materials[current_material]["texture"] = resized_texture

    def preprocess_texture(self, texture, max_size=64):
        """
        Resize texture to a manageable resolution while maintaining the aspect ratio.
        """
        h, w = texture.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            texture = cv2.resize(
                texture, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
        return texture


class OBJ:
    def __init__(self, obj_filename, mtl_filename=None, swapyz=False):
        """Parses a Wavefront OBJ file with .mtl material support."""
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.materials = {}
        self.current_material = None
        self.mtl = None

        # Ensure the obj file exists
        if not os.path.isfile(obj_filename):
            raise FileNotFoundError(f"OBJ file '{obj_filename}' not found.")

        with open(obj_filename, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                values = line.split()
                if not values:
                    continue
                if values[0] == "v":
                    v = list(map(float, values[1:4]))
                    if swapyz:
                        v = v[0], v[2], v[1]
                    self.vertices.append(v)
                elif values[0] == "vn":
                    v = list(map(float, values[1:4]))
                    if swapyz:
                        v = v[0], v[2], v[1]
                    self.normals.append(v)
                elif values[0] == "vt":
                    self.texcoords.append(list(map(float, values[1:3])))
                elif values[0] == "mtllib" and mtl_filename is not None:
                    self.mtl = MTL(mtl_filename)
                elif values[0] in ("usemtl", "usemat"):
                    self.current_material = values[1]
                elif values[0] == "f":
                    face = []
                    texcoords = []
                    norms = []
                    for v in values[1:]:
                        w = v.split("/")
                        face.append(int(w[0]) - 1)
                        if len(w) >= 2 and w[1]:
                            texcoords.append(int(w[1]) - 1)
                        else:
                            texcoords.append(-1)
                        if len(w) >= 3 and w[2]:
                            norms.append(int(w[2]) - 1)
                        else:
                            norms.append(-1)
                    self.faces.append((face, texcoords, norms, self.current_material))

    def get_material_color(self, material_name):
        """Returns the diffuse color of the material or a default color."""
        if self.mtl and material_name in self.mtl.materials:
            return self.mtl.materials[material_name].get(
                "Kd", [1.0, 1.0, 1.0]
            )  # Default white
        return [1.0, 1.0, 1.0]  # Default white
