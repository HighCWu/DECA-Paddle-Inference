import os
import cv2
import platform
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.utils.cpp_extension import load
curr_dir = os.path.dirname(__file__)
if platform.system() == 'Windows':
    standard_rasterize_cuda = \
        load(name='standard_rasterize_cuda',
             sources=[f'{curr_dir}/standard_rasterize_cuda.cc',
                      f'{curr_dir}/standard_rasterize_cuda_kernel.cu'],
             )
else:
    standard_rasterize_cuda = \
        load(name='standard_rasterize_cuda',
             sources=[f'{curr_dir}/standard_rasterize_cuda.cc',
                      f'{curr_dir}/standard_rasterize_cuda_kernel.cu'],
             )
standard_rasterize = standard_rasterize_cuda.standard_rasterize


def upsample_mesh(vertices, normals, faces, displacement_map, texture_map, dense_template):
    ''' Credit to Timo
    upsampling coarse mesh (with displacment map)
        vertices: vertices of coarse mesh, [nv, 3]
        normals: vertex normals, [nv, 3]
        faces: faces of coarse mesh, [nf, 3]
        texture_map: texture map, [256, 256, 3]
        displacement_map: displacment map, [256, 256]
        dense_template: 
    Returns: 
        dense_vertices: upsampled vertices with details, [number of dense vertices, 3]
        dense_colors: vertex color, [number of dense vertices, 3]
        dense_faces: [number of dense faces, 3]
    '''
    img_size = dense_template['img_size']
    dense_faces = dense_template['f']
    x_coords = dense_template['x_coords']
    y_coords = dense_template['y_coords']
    valid_pixel_ids = dense_template['valid_pixel_ids']
    valid_pixel_3d_faces = dense_template['valid_pixel_3d_faces']
    valid_pixel_b_coords = dense_template['valid_pixel_b_coords']

    pixel_3d_points = vertices[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
        vertices[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
        vertices[valid_pixel_3d_faces[:, 2], :] * \
        valid_pixel_b_coords[:, 2][:, np.newaxis]
    vertex_normals = normals
    pixel_3d_normals = vertex_normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
        vertex_normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
        vertex_normals[valid_pixel_3d_faces[:, 2], :] * \
        valid_pixel_b_coords[:, 2][:, np.newaxis]
    pixel_3d_normals = pixel_3d_normals / \
        np.linalg.norm(pixel_3d_normals, axis=-1)[:, np.newaxis]
    displacements = displacement_map[y_coords[valid_pixel_ids].astype(
        int), x_coords[valid_pixel_ids].astype(int)]
    dense_colors = texture_map[y_coords[valid_pixel_ids].astype(
        int), x_coords[valid_pixel_ids].astype(int)]
    offsets = np.einsum('i,ij->ij', displacements, pixel_3d_normals)
    dense_vertices = pixel_3d_points + offsets
    return dense_vertices, dense_colors, dense_faces

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py


def util_face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (paddle.arange(bs, dtype=paddle.int32) * nv)[:, None, None]
    vertices = vertices.reshape([bs * nv, 3])
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.astype(paddle.int64)]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    normals = paddle.zeros([bs * nv, 3])

    faces = faces + (paddle.arange(bs, dtype=paddle.int32)
                     * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape([bs * nv, 3])[faces.astype(paddle.int64)]

    faces = faces.reshape([-1, 3])
    vertices_faces = vertices_faces.reshape([-1, 3, 3])

    # paddle setitem bug, so use numpy instead
    # normals[faces[:, 1].astype(paddle.int64)] += paddle.cross(vertices_faces[:, 2] -
    #                                                           vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1])
    # normals[faces[:, 2].astype(paddle.int64)] += paddle.cross(vertices_faces[:, 0] -
    #                                                           vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2])
    # normals[faces[:, 0].astype(paddle.int64)] += paddle.cross(vertices_faces[:, 1] -
    #                                                           vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0])

    # v1 = paddle.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
    #                   vertices_faces[:, 0] - vertices_faces[:, 1])
    # v2 = paddle.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
    #                   vertices_faces[:, 1] - vertices_faces[:, 2])
    # v3 = paddle.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
    #                   vertices_faces[:, 2] - vertices_faces[:, 0])
    # normals_np = normals.numpy()
    # faces_np = faces.astype(paddle.int64).numpy()
    # normals_np[faces_np[:, 1]] += v1.numpy()
    # normals_np[faces_np[:, 2]] += v2.numpy()
    # normals_np[faces_np[:, 0]] += v3.numpy()
    # normals = paddle.to_tensor(normals_np)

    # paddle implement out different normal result, but get the same rendered imgs, unknow why
    faces = faces.astype(paddle.int64)
    normals = paddle.zeros([bs * nv, 3])
    v1 = paddle.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                      vertices_faces[:, 0] - vertices_faces[:, 1])
    v2 = paddle.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                      vertices_faces[:, 1] - vertices_faces[:, 2])
    v3 = paddle.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                      vertices_faces[:, 2] - vertices_faces[:, 0])
    normals = paddle.zeros_like(v1)
    normals = normals.scatter(faces[:, 1], normals[faces[:, 1]] + v1)
    normals = normals.scatter(faces[:, 2], normals[faces[:, 2]] + v2)
    normals = normals.scatter(faces[:, 0], normals[faces[:, 0]] + v3)
    normals = normals[:bs * nv]

    normals = F.normalize(normals, epsilon=1e-6, axis=1)
    normals = normals.reshape([bs, nv, 3])
    # pytorch only supports long and byte tensors for indexing
    return normals


def generate_triangles(h, w, margin_x=2, margin_y=5, mask=None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    # .
    # w*h
    triangles = []
    for x in range(margin_x, w-1-margin_x):
        for y in range(margin_y, h-1-margin_y):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:, [0, 2, 1]]
    return triangles


def tensor2image(tensor):
    image = tensor.detach().numpy()
    image = image*255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]]
    return image.astype(np.uint8).copy()

# borrowed from https://github.com/YadiraF/PRNet/blob/master/utils/write.py


def write_obj(obj_name,
              vertices,
              faces,
              colors=None,
              texture=None,
              uvcoords=None,
              uvfaces=None,
              inverse_face_order=False,
              normal_map=None,
              ):
    ''' Save 3D face model with texture. 
    Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        faces: shape = (ntri, 3)
        texture: shape = (uv_size, uv_size, 3)
        uvcoords: shape = (nver, 2) max value<=1
    '''
    if os.path.splitext(obj_name)[-1] != '.obj':
        obj_name = obj_name + '.obj'
    mtl_name = obj_name.replace('.obj', '.mtl')
    texture_name = obj_name.replace('.obj', '.png')
    material_name = 'FaceTexture'

    faces = faces.copy()
    # mesh lab start with 1, python/c++ start from 0
    faces += 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
        if uvfaces is not None:
            uvfaces = uvfaces[:, [2, 1, 0]]

    # write obj
    with open(obj_name, 'w') as f:
        # first line: write mtlib(material library)
        # f.write('# %s\n' % os.path.basename(obj_name))
        # f.write('#\n')
        # f.write('\n')
        if texture is not None:
            f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

        # write vertices
        if colors is None:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {}\n'.format(
                    vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(
                    vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(
                    faces[i, 2], faces[i, 1], faces[i, 0]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i, 0], uvcoords[i, 1]))
            f.write('usemtl %s\n' % material_name)
            # write f: ver ind/ uv ind
            uvfaces = uvfaces + 1
            for i in range(faces.shape[0]):
                f.write('f {}/{} {}/{} {}/{}\n'.format(
                    #  faces[i, 2], uvfaces[i, 2],
                    #  faces[i, 1], uvfaces[i, 1],
                    #  faces[i, 0], uvfaces[i, 0]
                    faces[i, 0], uvfaces[i, 0],
                    faces[i, 1], uvfaces[i, 1],
                    faces[i, 2], uvfaces[i, 2]
                )
                )
            # write mtl
            with open(mtl_name, 'w') as f:
                f.write('newmtl %s\n' % material_name)
                s = 'map_Kd {}\n'.format(
                    os.path.basename(texture_name))  # map to image
                f.write(s)

                if normal_map is not None:
                    name, _ = os.path.splitext(obj_name)
                    normal_name = f'{name}_normals.png'
                    f.write(f'disp {normal_name}')
                    # out_normal_map = normal_map / (np.linalg.norm(
                    #     normal_map, axis=-1, keepdims=True) + 1e-9)
                    # out_normal_map = (out_normal_map + 1) * 0.5

                    cv2.imwrite(
                        normal_name,
                        # (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
                        normal_map
                    )
            cv2.imwrite(texture_name, texture)


class Gather(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x, index):
        if self.dim < 0:
            self.dim += len(x.shape)
        x_range = list(range(len(x.shape)))
        x_range[0] = self.dim
        x_range[self.dim] = 0
        x_swaped = paddle.transpose(x, perm=x_range)
        index_range = list(range(len(index.shape)))
        index_range[0] = self.dim
        index_range[self.dim] = 0
        index_swaped = paddle.transpose(index, perm=index_range)
        dtype = index.dtype

        x_shape = paddle.shape(x_swaped)
        index_shape = paddle.shape(index_swaped)

        prod = paddle.cast(paddle.prod(x_shape), dtype=dtype) / x_shape[0]

        x_swaped_flattend = paddle.flatten(x_swaped)
        index_swaped_flattend = paddle.flatten(index_swaped)
        index_swaped_flattend *= prod

        bias = paddle.arange(start=0, end=prod, dtype=dtype)
        bias = paddle.reshape(bias, x_shape[1:])
        bias = paddle.crop(bias, index_shape[1:])
        bias = paddle.flatten(bias)
        bias = paddle.tile(bias, [index_shape[0]])
        index_swaped_flattend += bias

        gathered = paddle.index_select(
            x_swaped_flattend, index_swaped_flattend)
        gathered = paddle.reshape(gathered, index_swaped.shape)

        out = paddle.transpose(gathered, perm=x_range)

        return out


class StandardRasterizer(nn.Layer):
    """ Alg: https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation
    Notice:
        x,y,z are in image space, normalized to [-1, 1]
        can render non-squared image
        not differentiable
    """

    def __init__(self, height, width=None):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        if width is None:
            width = height
        self.h = h = height
        self.w = w = width

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        if h is None:
            h = self.h
        if w is None:
            w = self.h
        bz = vertices.shape[0]
        depth_buffer = paddle.zeros([bz, h, w]).astype(paddle.float32) + 1e6
        triangle_buffer = paddle.zeros([bz, h, w]).astype(paddle.int32) - 1
        baryw_buffer = paddle.zeros([bz, h, w, 3]).astype(paddle.float32)
        vert_vis = paddle.zeros([bz, vertices.shape[1]]).astype(paddle.float32)
        vertices = vertices.clone().astype(paddle.float32)
        #
        vertices[..., :2] = -vertices[..., :2]
        vertices[..., 0] = vertices[..., 0]*w/2 + w/2
        vertices[..., 1] = vertices[..., 1]*h/2 + h/2
        vertices[..., 0] = w - 1 - vertices[..., 0]
        vertices[..., 1] = h - 1 - vertices[..., 1]
        vertices[..., 0] = -1 + (2*vertices[..., 0] + 1)/w
        vertices[..., 1] = -1 + (2*vertices[..., 1] + 1)/h
        #
        vertices = vertices.clone().astype(paddle.float32)
        vertices[..., 0] = vertices[..., 0]*w/2 + w/2
        vertices[..., 1] = vertices[..., 1]*h/2 + h/2
        vertices[..., 2] = vertices[..., 2]*w/2
        f_vs = util_face_vertices(vertices, faces)

        standard_rasterize(f_vs, depth_buffer,
                           triangle_buffer, baryw_buffer, h, w)
        pix_to_face = triangle_buffer[:, :, :, None].astype(paddle.int32)
        bary_coords = baryw_buffer[:, :, :, None, :]
        vismask = (pix_to_face > -1).astype(paddle.float32)
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.reshape([
            attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1]])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face = pix_to_face * (1 - mask.astype(paddle.int64))
        idx = pix_to_face.reshape(
            [N * H * W * K, 1, 1]).expand([N * H * W * K, 3, D])
        pixel_face_vals = Gather(0)(
            attributes, idx).reshape([N, H, W, K, 3, D])
        pixel_vals = bary_coords[..., None] * pixel_face_vals
        pixel_vals = pixel_vals[..., 0, :] + \
            pixel_vals[..., 1, :] + pixel_vals[..., 2, :]
        # Replace masked values in output.
        pixel_vals = pixel_vals * (1 - mask[..., None].astype(paddle.float32))
        pixel_vals = pixel_vals[:, :, :, 0].transpose([0, 3, 1, 2])
        pixel_vals = paddle.concat(
            [pixel_vals, vismask[:, :, :, 0][:, None, :, :]], axis=1)
        return pixel_vals


class Render(nn.Layer):
    def __init__(self):
        super().__init__()

        self.rasterizer = StandardRasterizer(224)
        self.uv_rasterizer = StandardRasterizer(256)

        dense_triangles = generate_triangles(256, 256)
        self.register_buffer('dense_faces', paddle.to_tensor(
            dense_triangles).astype(paddle.int64)[None, :, :])

        # SH factors for lighting
        pi = np.pi
        self.constant_factor = paddle.to_tensor([1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),
                                                 ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3) *
                                                 (np.sqrt(5/(12*pi))), (pi/4) *
                                                 (3)*(np.sqrt(5/(12*pi))),
                                                 (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi)))]).astype(paddle.float32)

    @property
    def raw_uvcoords(self):
        uvcoords = self.uvcoords.clone()
        uvcoords[...,1] = -uvcoords[...,1]
        uvcoords = (uvcoords + 1)/2
        uvcoords = uvcoords[...,:-1]

        return uvcoords


    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point'):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights: 
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]
        # rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10
        # attributes
        face_vertices = util_face_vertices(
            vertices, self.faces.expand([batch_size, -1, -1]))
        normals = vertex_normals(
            vertices, self.faces.expand([batch_size, -1, -1]))
        face_normals = util_face_vertices(
            normals, self.faces.expand([batch_size, -1, -1]))
        transformed_normals = vertex_normals(
            transformed_vertices, self.faces.expand([batch_size, -1, -1]))
        transformed_face_normals = util_face_vertices(
            transformed_normals, self.faces.expand([batch_size, -1, -1]))

        attributes = paddle.concat([self.face_uvcoords.expand([batch_size, -1, -1, -1]),
                                    transformed_face_normals.detach(),
                                    face_vertices.detach(),
                                    face_normals],
                                   -1)
        # rasterize
        rendering = self.rasterizer(
            transformed_vertices, self.faces.expand([batch_size, -1, -1]), attributes)

        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).transpose([0, 2, 3, 1])[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (
            transformed_normal_map[:, 2:, :, :] < -0.05).astype(paddle.float32)

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type == 'point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.transpose([0, 2, 3, 1]).reshape(
                        [batch_size, -1, 3]), normal_images.transpose([0, 2, 3, 1]).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).transpose([0, 3, 1, 2])
                else:
                    shading = self.add_directionlight(normal_images.transpose([
                        0, 2, 3, 1]).reshape([batch_size, -1, 3]), lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).transpose([0, 3, 1, 2])
            images = albedo_images*shading_images
        else:
            images = albedo_images
            shading_images = images.detach()*0.

        outputs = {
            'images': images*alpha_images,
            'albedo_images': albedo_images*alpha_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images*alpha_images,
            'transformed_normals': transformed_normals,
        }

        return outputs

    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = paddle.stack([
            N[:, 0]*0.+1., N[:, 0], N[:, 1],
            N[:, 2], N[:, 0]*N[:, 1], N[:, 0]*N[:, 2],
            N[:, 1]*N[:, 2], N[:, 0]**2 - N[:, 1]**2, 3*(N[:, 2]**2) - 1
        ],
            1)  # [bz, 9, h, w]
        sh = sh*self.constant_factor[None, :, None, None]
        # [bz, 9, 3, h, w]
        shading = paddle.sum(
            sh_coeff[:, :, :, None, None]*sh[:, :, None, :, :], 1)
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(
            light_positions[:, :, None, :] - vertices[:, None, :, :], axis=3)
        # normals_dot_lights = paddle.clip((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (
            normals[:, None, :, :]*directions_to_lights).sum(axis=3)
        shading = normals_dot_lights[:, :, :,
                                     None]*light_intensities[:, :, None, :]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:, :, :3]
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(
            light_direction[:, :, None, :].expand([-1, -1, normals.shape[1], -1]), axis=3)
        # normals_dot_lights = paddle.clip((normals[:,None,:,:]*directions_to_lights).sum(axis=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(axis=3)
        normals_dot_lights = paddle.clip(
            (normals[:, None, :, :]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:, :, :,
                                     None]*light_intensities[:, :, None, :]
        return shading.mean(1)

    def world2uv(self, vertices):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = util_face_vertices(
            vertices, self.faces.expand([batch_size, -1, -1]))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand([
            batch_size, -1, -1]), self.uvfaces.expand([batch_size, -1, -1]), face_vertices)[:, :3]
        return uv_vertices


class DecaInference(nn.Layer):
    def __init__(self):
        super().__init__()
        fixed_dis = np.load('fixed_displacement_256.npy')
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(
            'texture_data_256.npy', allow_pickle=True, encoding='latin1').item()
        self.fixed_uv_dis = paddle.to_tensor(fixed_dis).astype(paddle.float32)
        self.render = Render()

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()

        uv_z = uv_z*self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + \
            self.fixed_uv_dis[None, None, :, :]*uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.transpose(
            [0, 2, 3, 1]).reshape([batch_size, -1, 3])
        uv_detail_normals = vertex_normals(
            dense_vertices, self.render.dense_faces.expand([batch_size, -1, -1]))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).transpose([0, 3, 1, 2])
        uv_detail_normals = uv_detail_normals*self.uv_face_eye_mask + \
            uv_coarse_normals*(1-self.uv_face_eye_mask)
        return uv_detail_normals

    @paddle.no_grad()
    def render_images(self, codedict, opdict, visdict, use_detail=True, use_tex=True):
        self.render.faces = visdict['faces']
        self.render.uvcoords = visdict['uvcoords']
        self.render.uvfaces = visdict['uvfaces']
        self.render.face_uvcoords = util_face_vertices(
            self.render.uvcoords, self.render.uvfaces)
        self.uv_face_eye_mask = visdict['uv_face_eye_mask']

        verts = opdict['verts']
        trans_verts = visdict['trans_verts']
        albedo = opdict['albedo']
        if use_detail:
            uv_z = opdict['uv_z']
            normals = opdict['normals']
            uv_detail_normals = self.displacement2normal(uv_z, verts, normals)
            uv_shading = self.render.add_SHlight(
                uv_detail_normals, codedict['light'])
            uv_texture = albedo*uv_shading

            opdict['uv_texture'] = uv_texture
            opdict['uv_detail_normals'] = uv_detail_normals
            opdict['displacement_map'] = uv_z + \
                self.fixed_uv_dis[None, None, :, :]
        uv_pverts = self.render.world2uv(trans_verts)
        uv_gt = F.grid_sample(visdict['images'], uv_pverts.transpose(
            [0, 2, 3, 1])[:, :, :, :2], mode='bilinear')
        if use_tex:
            # TODO: poisson blending should give better-looking results
            uv_texture_gt = uv_gt[:, :3, :, :]*self.uv_face_eye_mask + \
                (opdict['uv_texture'][:, :3, :, :]*(1-self.uv_face_eye_mask))
        else:
            uv_texture_gt = uv_gt[:, :3, :, :]*self.uv_face_eye_mask + \
                (paddle.ones_like(uv_gt[:, :3, :, :])
                 * (1-self.uv_face_eye_mask)*0.7)
        uv_texture_gt = opdict['uv_texture_gt'] = opdict.get('uv_texture_gt', uv_texture_gt)
        return self.render(verts, trans_verts, uv_texture_gt, codedict['light'])['images']

    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
        write_obj(filename, vertices, faces,
                  texture=texture,
                  uvcoords=uvcoords,
                  uvfaces=uvfaces,
                  normal_map=normal_map)
        # upsample mesh, save detailed mesh
        texture = texture[:, :, [2, 1, 0]]
        normals = opdict['normals'][i].cpu().numpy()
        displacement_map = opdict['displacement_map'][i].cpu(
        ).numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = upsample_mesh(
            vertices, normals, faces, displacement_map, texture, self.dense_template)
        write_obj(filename.replace('.obj', '_detail.obj'),
                  dense_vertices,
                  dense_faces,
                  colors=dense_colors,
                  inverse_face_order=True)
