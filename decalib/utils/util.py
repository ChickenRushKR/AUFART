# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import numpy as np
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict
import os
from scipy.ndimage import morphology
from skimage.io import imsave
import cv2
import torchvision

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
                    vertices[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]

    vertex_normals = normals
    pixel_3d_normals = vertex_normals[valid_pixel_3d_faces[:, 0], :] * valid_pixel_b_coords[:, 0][:, np.newaxis] + \
                    vertex_normals[valid_pixel_3d_faces[:, 1], :] * valid_pixel_b_coords[:, 1][:, np.newaxis] + \
                    vertex_normals[valid_pixel_3d_faces[:, 2], :] * valid_pixel_b_coords[:, 2][:, np.newaxis]
    pixel_3d_normals = pixel_3d_normals / np.linalg.norm(pixel_3d_normals, axis=-1)[:, np.newaxis]
    displacements = displacement_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    dense_colors = texture_map[y_coords[valid_pixel_ids].astype(int), x_coords[valid_pixel_ids].astype(int)]
    offsets = np.einsum('i,ij->ij', displacements, pixel_3d_normals)
    dense_vertices = pixel_3d_points + offsets
    write_obj('pixel_3d_points.obj',
              pixel_3d_points,
              dense_faces,
              colors=dense_colors,
              inverse_face_order=True)
    return dense_vertices, dense_colors, dense_faces

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
                f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        else:
            for i in range(vertices.shape[0]):
                f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

        # write uv coords
        if texture is None:
            for i in range(faces.shape[0]):
                f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
        else:
            for i in range(uvcoords.shape[0]):
                f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
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
                s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
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


## load obj,  similar to load_obj from pytorch3d
def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = torch.tensor(verts, dtype=torch.float32)
    uvcoords = torch.tensor(uvcoords, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.long); faces = faces.reshape(-1, 3) - 1
    uv_faces = torch.tensor(uv_faces, dtype=torch.long); uv_faces = uv_faces.reshape(-1, 3) - 1
    return (
        verts,
        uvcoords,
        faces,
        uv_faces
    )

# ---------------------------- process/generate vertices, normals, faces
def generate_triangles(h, w, margin_x=2, margin_y=5, mask = None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    #.
    # w*h
    triangles = []
    for x in range(margin_x, w-1-margin_x):
        for y in range(margin_y, h-1-margin_y):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:,[0,2,1]]
    return triangles

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
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
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]
    
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
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn
    
# -------------------------------------- image processing
# borrowed from: https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters
def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()

def get_gaussian_kernel(kernel_size: int, sigma: float):
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("kernel_size must be an odd positive integer. "
                        "Got {}".format(kernel_size))
    window_1d = gaussian(kernel_size, sigma)
    return window_1d

def get_gaussian_kernel2d(kernel_size, sigma):
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of length two. Got {}"
                        .format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d

def gaussian_blur(x, kernel_size=(3,3), sigma=(0.8,0.8)):
    b, c, h, w = x.shape
    kernel = get_gaussian_kernel2d(kernel_size, sigma).to(x.device).to(x.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = [(k - 1) // 2 for k in kernel_size]
    return F.conv2d(x, kernel, padding=padding, stride=1, groups=c)

def _compute_binary_kernel(window_size):
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])

def median_blur(x, kernel_size=(3,3)):
    b, c, h, w = x.shape
    kernel = _compute_binary_kernel(kernel_size).to(x.device).to(x.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = [(k - 1) // 2 for k in kernel_size]
    features = F.conv2d(x, kernel, padding=padding, stride=1, groups=c)
    features = features.view(b,c,-1,h,w)
    median = torch.median(features, dim=2)[0]
    return median

def get_laplacian_kernel2d(kernel_size: int):
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (int): filter size should be odd.

    Returns:
        Tensor: 2D tensor with laplacian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_laplacian_kernel2d(3)
        tensor([[ 1.,  1.,  1.],
                [ 1., -8.,  1.],
                [ 1.,  1.,  1.]])

        >>> kornia.image.get_laplacian_kernel2d(5)
        tensor([[  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1., -24.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.],
                [  1.,   1.,   1.,   1.,   1.]])

    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d

def laplacian(x):
    # https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/filters/laplacian.html
    b, c, h, w = x.shape
    kernel_size = 3
    kernel = get_laplacian_kernel2d(kernel_size).to(x.device).to(x.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = (kernel_size - 1) // 2
    return F.conv2d(x, kernel, padding=padding, stride=1, groups=c)

def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [batch_size, 3] tensor containing X, Y, and Z angles.
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [batch_size, 3, 3]. rotation matrices.
    '''
    angles = angles*(np.pi)/180.
    s = torch.sin(angles)
    c = torch.cos(angles)

    cx, cy, cz = (c[:, 0], c[:, 1], c[:, 2])
    sx, sy, sz = (s[:, 0], s[:, 1], s[:, 2])

    zeros = torch.zeros_like(s[:, 0]).to(angles.device)
    ones = torch.ones_like(s[:, 0]).to(angles.device)

    # Rz.dot(Ry.dot(Rx))
    R_flattened = torch.stack(
    [
      cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
      sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
          -sy,                cy * sx,                cy * cx,
    ],
    dim=0) #[batch_size, 9]
    R = torch.reshape(R_flattened, (-1, 3, 3)) #[batch_size, 3, 3]
    return R

def binary_erosion(tensor, kernel_size=5):
    # tensor: [bz, 1, h, w]. 
    device = tensor.device
    mask = tensor.cpu().numpy()
    structure=np.ones((kernel_size,kernel_size))
    new_mask = mask.copy()
    for i in range(mask.shape[0]):
        new_mask[i,0] = morphology.binary_erosion(mask[i,0], structure)
    return torch.from_numpy(new_mask.astype(np.float32)).to(device)

def flip_image(src_image, kps):
    '''
        purpose:
            flip a image given by src_image and the 2d keypoints
        flip_mode: 
            0: horizontal flip
            >0: vertical flip
            <0: horizontal & vertical flip
    '''
    h, w = src_image.shape[0], src_image.shape[1]
    src_image = cv2.flip(src_image, 1)
    if kps is not None:
        kps[:, 0] = w - 1 - kps[:, 0]
        kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
        kps[:, :] = kps[kp_map]

    return src_image, kps

# -------------------------------------- io
def copy_state_dict(cur_state_dict, pre_state_dict, prefix='', load_name=None):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    for k in cur_state_dict.keys():
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        try:
            if v is None:
                # print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            # print('copy param {} failed'.format(k))
            continue

def check_mkdir(path):
    if not os.path.exists(path):
        print('creating %s' % path)
        os.makedirs(path)

def check_mkdirlist(pathlist):
    for path in pathlist:
        if not os.path.exists(path):
            print('creating %s' % path)
            os.makedirs(path)

def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image*255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)[:,:,[2,1,0]]
    return image.astype(np.uint8).copy()

def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

# original saved file with DataParallel
def remove_module(state_dict):
# create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def dict_tensor2npy(tensor_dict):
    npy_dict = {}
    for key in tensor_dict:
        npy_dict[key] = tensor_dict[key][0].cpu().numpy()
    return npy_dict
        
# ---------------------------------- visualization
end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1])/200), 1)
    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1]==4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), radius)
        image = cv2.circle(image,(int(st[0]), int(st[1])), radius, c, radius*2)  

    return image

def plot_verts(image, kpts, color = 'r'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (0, 0, 255)
    elif color == 'y':
        c = (0, 255, 255)
    image = image.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(image,(int(st[0]), int(st[1])), 2, c, -1)  
        # image = cv2.putText(image, f'{i}', (int(st[0]), int(st[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1, cv2.LINE_AA)  

    return image

def tensor_vis_landmarks(images, landmarks, gt_landmarks=None, color = 'g', isScale=True):
    # visualize landmarks
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1,2,0)[:,:,[2,1,0]].copy(); image = (image*255)
        if isScale:
            predicted_landmark = predicted_landmarks[i]
            predicted_landmark[...,0] = predicted_landmark[...,0]*image.shape[1]/2 + image.shape[1]/2
            predicted_landmark[...,1] = predicted_landmark[...,1]*image.shape[0]/2 + image.shape[0]/2
        else:
            predicted_landmark = predicted_landmarks[i]
        if predicted_landmark.shape[0] == 68:
            image_landmarks = plot_kpts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks, gt_landmarks_np[i]*image.shape[0]/2 + image.shape[0]/2, 'r')
        else:
            image_landmarks = plot_verts(image, predicted_landmark, color)
            if gt_landmarks is not None:
                image_landmarks = plot_verts(image_landmarks, gt_landmarks_np[i]*image.shape[0]/2 + image.shape[0]/2, 'r')
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(vis_landmarks[:,:,:,[2,1,0]].transpose(0,3,1,2))/255.#, dtype=torch.float32)
    return vis_landmarks


############### for training
def load_local_mask(image_size=256, mode='bbx'):
    if mode == 'bbx':
        # UV space face attributes bbx in size 2048 (l r t b)
        # face = np.array([512, 1536, 512, 1536]) #
        face = np.array([400, 1648, 400, 1648])
        # if image_size == 512:
            # face = np.array([400, 400+512*2, 400, 400+512*2])
            # face = np.array([512, 512+512*2, 512, 512+512*2])

        forehead = np.array([550, 1498, 430, 700+50])
        eye_nose = np.array([490, 1558, 700, 1050+50])
        mouth = np.array([574, 1474, 1050, 1550])
        ratio = image_size / 2048.
        face = (face * ratio).astype(np.int)
        forehead = (forehead * ratio).astype(np.int)
        eye_nose = (eye_nose * ratio).astype(np.int)
        mouth = (mouth * ratio).astype(np.int)
        regional_mask = np.array([face, forehead, eye_nose, mouth])

    return regional_mask

def visualize_grid(visdict, savepath=None, size=224, dim=1, return_gird=True):
    '''
    image range should be [0,1]
    dim: 2 for horizontal. 1 for vertical
    '''
    assert dim == 1 or dim==2
    grids = {}
    for key in visdict:
        _,_,h,w = visdict[key].shape
        if dim == 2:
            new_h = size; new_w = int(w*size/h)
        elif dim == 1:
            new_h = int(h*size/w); new_w = size
        grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu())
    grid = torch.cat(list(grids.values()), dim)
    grid_image = (grid.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
    if savepath:
        cv2.imwrite(savepath, grid_image)
    if return_gird:
        return grid_image

def map_landmark_to_action_unit():
    # Define the mapping between landmark index and action unit index
    # Format: {action_unit_index: [list of landmark indices]}
    mapping = {
        1: [22, 23],
        2: [18, 27],
        4: [[22, 23]],
        5: [38, 45],
        6: [3, 15],
        7: [41, 48],
        9: [28],
        10: [51, 53],
        12: [49, 55],
        15: [60, 56],
        16: [[63, 67]],
        18: [[63, 67]],
        20: [[63, 67]],
        22: [[63, 67]],
        23: [[63, 67]],
        24: [[63, 67]],
        25: [[63, 67]],
        26: [8, 10],
        27: [[63, 67]],
        32: [[63, 67]],
        38: [31],
        39: [31],
    }
    return mapping

def draw_activation_circles(images, kpts, aus, au_weight, params=None):
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (0, 0, 255)  # 빨간색 (BGR 형식)
    images = images.detach().cpu().numpy()
    kpts = kpts.detach().cpu().numpy()
    vis_landmarks = []
    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1,2,0)[:,:,[2,1,0]].copy(); image = (image*255)
        kpt = kpts[i]*image.shape[0]/2 + image.shape[0]/2
        for j in range(kpts.shape[1]):
            related_au = au_weight.lmk_to_au(j)
            st = kpt[j, :2]
            if True in (0.5 <= aus[i,related_au]):
                image = cv2.circle(image,(int(st[0]), int(st[1])), 1, green, 2)
            else:
                image = cv2.circle(image,(int(st[0]), int(st[1])), 1, red, 2)
        if params != None:
            text = f"exp: {torch.linalg.norm(torch.abs(params['exp'][i])).item():.2f} / jaw: {torch.linalg.norm(torch.abs(params['jaw'][i][3:])).item():.2f}"
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_position = (image.shape[1] - text_size[0] - 10, image.shape[0] - 10)
            image = cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)
        vis_landmarks.append(image)
    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(vis_landmarks[:,:,:,[2,1,0]].transpose(0,3,1,2))/255.#, dtype=torch.float32)
    return vis_landmarks

def vis_au(srcAU, dstAU, image_size=448):

    # AU 라벨 리스트
    au_labels = ["AU1", "AU2", "AU4", "AU5", "AU7", "AU9",
                 "AU10", "AU12", "AU15", "AU20", "AU23", "AU26",
                 "AU1", "AU2", "AU4", "AU5", "AU7", "AU9",
                 "AU10", "AU12", "AU15", "AU20", "AU23", "AU26"]

    # AU 라벨을 이미지에 쓰기
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    num_rows, num_cols = 4, 6
    cell_width = image_size // num_cols
    cell_height = image_size // num_rows
    compare = [0,1,2,3,5,6,7,9,12,17,19,22]
    imgs = []

    # 이미지 초기화 (흰 배경)
    for i in range(srcAU.shape[0]):
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        if srcAU.shape[0] > 12:
            concated = torch.cat([srcAU[i][compare], dstAU[i][compare]], dim=0)
        else:
            concated = torch.cat([srcAU[i], dstAU[i]], dim=0)
        # srcAU와 dstAU의 결과를 이미지에 쓰기
        for j in range(num_rows):
            for k in range(num_cols):
                cell_text = au_labels[j * num_cols + k]
                text_size = cv2.getTextSize(cell_text, font, font_scale, font_thickness)[0]
                text_x = k * cell_width + (cell_width - text_size[0]) // 2
                text_y = j * cell_height + (cell_height + text_size[1]) // 2
                color = (255, 0, 0) if concated[j * num_cols + k] >= 0.5 else (0, 0, 0)  # 파란색 또는 검은색
                fill_color = (255, 200, 200) if concated[j * num_cols + k] >= 0.5 else (255, 255, 255)
                cell_rect = ((k * cell_width, j * cell_height), ((k + 1) * cell_width, (j + 1) * cell_height))
                cv2.rectangle(img, cell_rect[0], cell_rect[1], fill_color, cv2.FILLED)
                # 이미지에 글자 삽입
                cv2.putText(img, cell_text, (text_x, text_y), font, font_scale, color, font_thickness)
                
        
        cv2.line(img, (0, image_size // 2), (image_size, image_size // 2),
                     (0, 0, 0), 1)
        imgs.append(img)

    imgs = np.stack(imgs)
    vis_au = torch.from_numpy(imgs[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2)) / 255.
    return vis_au

def visualize_arrays(images, au_gt, au_pred, image_size=(224, 224)):
# Function to create the visualization
# def visualize_arrays(array1, array2, image_size=(224, 224)):
    # Create an empty image array
    imgsize = images.shape[-1]
    vis_au = []
    au_gt = au_gt.detach().cpu().numpy()
    au_pred = au_pred.detach().cpu().numpy()
    for i in range(images.shape[0]):
        # image = images[i]
        image = np.zeros((imgsize, imgsize, 3), dtype=np.uint8)

        # Create tensors for array1 and array2 visualizations
        # array1_row = au_gt[i]#.view(41, -1)
        # array2_row = au_pred[i]#.view(41, -1)
        array1_row = au_gt[i] * 255
        array2_row = au_pred[i] * 255

        # Normalize array values to fit in the [0, 255] range
        # array1_normalized = (F.normalize(array1_row, p=2, dim=1) * 255).int().numpy()
        # array2_normalized = (F.normalize(array2_row, p=2, dim=1) * 255).int().numpy()

        # Fill the corresponding rows in the image array
        index_row = np.ones((imgsize//3, imgsize, 3), dtype=np.uint8) * 255  # Index row in white
        image[:imgsize//3, :, :] = index_row

        # Display black text in the center of each box in the index row
        for i in range(41):
            index_text = str(i)
            x_position = int(i * (imgsize / 41) + (imgsize / (2 * 41)))
            y_position = int(imgsize / 6)
            cv2.putText(image, index_text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 0), 1, cv2.LINE_AA)

            image[imgsize//3:2*imgsize//3, int(i*(imgsize/41)):int((i+1)*(imgsize/41)), 1] = array1_row[i] #array1_normalized.squeeze()  # Array1 visualization in green channel
            image[2*imgsize//3:, int(i*(imgsize/41)):int((i+1)*(imgsize/41)), 1] = array2_row[i]
            # image[2*imgsize//3:, :, 1] = array2_row #array2_normalized.squeeze()  # Array2 visualization in blue channel
        vis_au.append(image)
    vis_au = np.stack(vis_au)
    vis_au = torch.from_numpy(vis_au[:,:,:,[2,1,0]].transpose(0,3,1,2))/255.#, dtype=torch.float32)
    return vis_au
    

class au_weights:
    def __init__(self): 
        # landmark index
        # brow: 0~19
        self.brow = [i for i in range(0, 20)]

        self.brow_inner = [1, 3, 5, 6, 8, 9, 11, 13, 15, 16, 18, 19]
        self.brow_outer = [0, 1, 2, 4, 7, 8, 10, 11, 12, 14, 17, 18]
        # eye_low: 21, 22, 28, 29, 30, 31, 32, 34, 35, 37, 38, 44, 45, 46, 47, 48, 50, 51
        self.eye_low = [21, 22, 28, 29, 30, 31, 32, 34, 35, 37, 38, 44, 45, 46, 47, 48, 50, 51]
        # eye_up: 20, 21, 22, 23, 24, 25, 26, 27, 33, 36, 37, 38, 39, 40, 41, 42, 43, 49
        self.eye_up = [20, 21, 22, 23, 24, 25, 26, 27, 33, 36, 37, 38, 39, 40, 41, 42, 43, 49]
        # eye_all: 20~51
        self.eye_all = [i for i in range(20, 52)]
        # nose: 52~64
        self.nose = [i for i in range(52, 65)]
        # lip_up: 65, 66, 69, 70, 71, 72, 73, 74, 75, 76, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 103, 104
        self.lip_up = [65, 66, 69, 70, 71, 72, 73, 74, 75, 76, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 103, 104]
        # lip_end: 71, 72, 73, 74, 79, 80, 81, 82, 85, 86, 88, 89, 90, 91, 92, 93, 97, 98, 99, 100, 103, 104
        self.lip_end = [71, 72, 73, 74, 79, 80, 81, 82, 85, 86, 88, 89, 90, 91, 92, 93, 97, 98, 99, 100, 103, 104]
        # mouth: 65~104
        self.mouth = [i for i in range(65, 105)]
        self.lip_out = [65,68,69,70,71,72,77,80,82,84,85,87,88,89,90,95,98,100,102,103]

        # AU index
        self.brow_au = [2, 31, 32]
        self.brow_inner_au = [0, 27, 28]
        self.brow_outer_au = [1, 29, 30]

        self.eye_low_au = [3]
        self.eye_up_au = [4, 33, 34]
        self.eye_all_au = [5]
        self.nose_au = [6, 8, 25, 26, 37, 38]
        self.lip_up_au = [7, 8, 35, 36, 37, 38]
        self.lip_end_au = [9, 10, 11, 12, 13, 39, 40]
        self.mouth_au = [i for i in range(14, 25)]

        # AU landmarks involve
        self.inner_brow_raiser = [[21,22,17,26],[39,42,36,45]]
        self.outer_brow_raiser = [[19,37,23,43],[20,38,24,44]]
        self.brow_lowerer = [[21],[22]]
        self.upper_lid_raiser = [[37,40,43,46],[38,41,24,44]]
        self.lid_tightener = [[37,40,43,46],[38,41,24,44]]
        self.nose_wrinkler = [[27],[29]]
        self.upper_lip_raiser = [[60,62,32,33,34,41,46],[65,63,50,51,52,48,54]]
        self.lip_corner_puller = [[49,39,40,41,42,46,47],[54,48,48,48,54,54,54]]
        self.lip_corner_depressor = [[39,40,41,42,46,47],[48,48,48,54,54,54]]
        self.lip_stretcher = [[48],[54]]
        self.lip_tightener = [[51],[57]]
        self.jaw_drop = [[48,51,50,52],[54,57,58,56]]

    def au_related_landmark_distance(self, lmk):
        inner_brow_raiser = self.distance(lmk[:,self.inner_brow_raiser[0],:], lmk[:,self.inner_brow_raiser[1],:])
        outer_brow_raiser = self.distance(lmk[:,self.outer_brow_raiser[0],:], lmk[:,self.outer_brow_raiser[1],:])
        brow_lowerer = self.distance(lmk[:,self.brow_lowerer[0],:], lmk[:,self.brow_lowerer[1],:])
        upper_lid_raiser = self.distance(lmk[:,self.upper_lid_raiser[0],:], lmk[:,self.upper_lid_raiser[1],:])
        lid_tightener = self.distance(lmk[:,self.lid_tightener[0],:], lmk[:,self.lid_tightener[1],:])
        nose_wrinkler = self.distance(lmk[:,self.nose_wrinkler[0],:], lmk[:,self.nose_wrinkler[1],:])
        upper_lip_raiser = self.distance(lmk[:,self.upper_lip_raiser[0],:], lmk[:,self.upper_lip_raiser[1],:])
        lip_corner_puller = self.distance(lmk[:,self.lip_corner_puller[0],:], lmk[:,self.lip_corner_puller[1],:])
        lip_corner_depressor = self.distance(lmk[:,self.lip_corner_depressor[0],:], lmk[:,self.lip_corner_depressor[1],:])
        lip_stretcher = self.distance(lmk[:,self.lip_stretcher[0],:], lmk[:,self.lip_stretcher[1],:])
        lip_tightener = self.distance(lmk[:,self.lip_tightener[0],:], lmk[:,self.lip_tightener[1],:])
        jaw_drop = self.distance(lmk[:,self.jaw_drop[0],:], lmk[:,self.jaw_drop[1],:])
        return inner_brow_raiser, outer_brow_raiser, brow_lowerer, upper_lid_raiser, lid_tightener, nose_wrinkler, upper_lip_raiser,\
            lip_corner_puller, lip_corner_depressor, lip_stretcher, lip_tightener, jaw_drop

    def lmk_to_au(self, idx):
        related_au = []
        if idx in self.brow:
            related_au += self.brow_au
        if idx in self.brow_inner:
            related_au += self.brow_inner_au
        if idx in self.brow_outer:
            related_au += self.brow_outer_au
        if idx in self.brow:
            related_au += self.brow_au
        if idx in self.eye_low:
            related_au += self.eye_low_au
        if idx in self.eye_up:
            related_au += self.eye_up_au
        if idx in self.eye_all:
            related_au += self.eye_all_au
        if idx in self.nose:
            related_au += self.nose_au
        if idx in self.lip_up:
            related_au += self.lip_up_au
        if idx in self.lip_end:
            related_au += self.lip_end_au
        if idx in self.mouth:
            related_au += self.mouth_au
        return list(set(related_au))  


    def distance(self, a, b):
        return torch.sqrt(((a - b)**2).sum(2))

        # for i in range(landmarks_gt.shape[0]):
        #     if 1.0 in au[i,au_weight.brow_au]:
        #         weights[i,au_weight.brow] = 3.0
        #     if 1.0 in au[i,au_weight.eye_up_au]:
        #         weights[i,au_weight.eye_up] = 3.0
        #     if 1.0 in au[i,au_weight.eye_low_au]:
        #         weights[i,au_weight.eye_low] = 3.0
        #     if 1.0 in au[i,au_weight.eye_all_au]:
        #         weights[i,au_weight.eye_all] = 3.0
        #     if 1.0 in au[i,au_weight.nose_au]:
        #         weights[i,au_weight.nose] = 3.0
        #     if 1.0 in au[i,au_weight.lip_up_au]:
        #         weights[i,au_weight.lip_up] = 3.0
        #     if 1.0 in au[i,au_weight.lip_end_au]:
        #         weights[i,au_weight.lip_end] = 3.0
        #     if 1.0 in au[i,au_weight.mouth_au]:
        #         weights[i,au_weight.mouth] = 3.0
                

# def draw_activation_circles(face_images, landmarks_batch, action_unit_predictions_batch, au_weight):
#     # Copy the face images to avoid modifying the original images
#     vis_aus = []
#     face_images = face_images.detach().cpu().numpy()
#     # result_images = face_images.clone().permute(0,2,3,1).detach().cpu().numpy(); 
#     # result_images=(result_images*255).astype(np.uint8).copy();
#     # result_images=result_images[:,:,[2,1,0]]
#     # image.transpose[:,:,[2,1,0]].copy(); image = (image*255)
#     landmarks_batch = landmarks_batch.clone().detach().cpu().numpy()
#     # action_unit_predictions_batch = np.asarray(action_unit_predictions_batch)
#     # Map action units to landmarks
#     action_unit_mapping = map_landmark_to_action_unit()
#     au_mouth = {15:False,17:False,19:False,21:False,22:False,23:False,24:False,26:False,31:False}
#     for batch_idx in range(len(face_images)):
#         image = face_images[batch_idx]
#         image = image.transpose(1,2,0)[:,:,[2,1,0]].copy(); image = (image*255)
#         for action_unit, landmark_indices in action_unit_mapping.items():
#             landmark_indices = np.asarray(landmark_indices)
#             # Get coordinates
#             if len(landmark_indices.shape) == 1:
#                 coords = [landmarks_batch[batch_idx, i - 1] for i in landmark_indices]
#             elif len(landmark_indices.shape) == 2 and landmark_indices.shape[0] == 1:
#                 coords = [np.mean([landmarks_batch[batch_idx, i - 1] for i in landmark_indices[0]], axis=0)]
#             else:
#                 coords = np.mean([landmarks_batch[batch_idx, i - 1] for i in landmark_indices], axis=1)
#             # Check if any of the mapped landmarks' action unit is activated
#             activation = action_unit_predictions_batch[batch_idx, action_unit - 1] >= 0.5

#             if action_unit in au_mouth:
#                 au_mouth[action_unit] = activation
#                 if True in au_mouth:
#                     activation = True
#             # Draw circle based on activation status
#             color = (255, 0, 0) if not activation else (0, 255, 0)

#             for coord in coords:
#                 coord = (coord + 1) * 112
#                 cv2.circle(image, (int(coord[0]), int(coord[1])), 4, color, -1)
    
#         vis_aus.append(image)

#     result_images = np.stack(vis_aus)
#     result_images = torch.from_numpy(result_images[:,:,:,[2,1,0]].transpose(0,3,1,2))/255.#, dtype=torch.float32)
#     return result_images

