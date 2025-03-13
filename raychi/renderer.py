import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from math import pi, tan
import time
import json
import sys  # Added import

# --- Initialize Taichi
ti.init(arch=ti.gpu, debug=False)
print("Running on:", ti.cfg.arch)

# --- Global Parameters as Taichi Fields ---
ENABLE_AO = ti.field(ti.u1, shape=())
ENABLE_DIRECT_LIGHTING = ti.field(ti.u1, shape=())
RR_prob = ti.field(ti.f32, shape=())
max_ray_pool_param = ti.field(ti.i32, shape=())  # New field for parameter
max_ambient_requests_param = ti.field(ti.i32, shape=())  # New field for parameter

@ti.dataclass
class SceneObject:
    type: ti.i32                              # 0 = plane, 1 = sphere
    center: ti.types.vector(3, ti.f32)         # For spheres
    radius: ti.f32                           # For spheres
    normal: ti.types.vector(3, ti.f32)         # For planes
    offset: ti.f32                           # For planes
    diffuse: ti.types.vector(3, ti.f32)        # Albedo or emission
    specular: ti.types.vector(3, ti.f32)       # Specular color (for metal)
    ior: ti.f32                              # Index of refraction (dielectric)
    do_checkboard: ti.i32                    # (Unused here)
    scale: ti.i32                            # (Unused here)
    fuzz: ti.f32                             # For metal materials
    material: ti.i32                         # 0: Emissive, 1: Diffuse, 2: Metal, 3: Dielectric, 4: Fuzz Metal

@ti.dataclass
class Ray:
    pixel_index: ti.i32
    origin: ti.types.vector(3, ti.f32)
    direction: ti.types.vector(3, ti.f32)
    depth: ti.i32
    weight: ti.types.vector(3, ti.f32)
    rt: ti.i32        # 0: primary, 1: AO
    ambient_req_id: ti.i32  # -1 if not applicable

@ti.dataclass
class AmbientRequest:
    pixel_index: ti.i32
    ambient_contrib: ti.types.vector(3, ti.f32)
    occluded: ti.i32
    total: ti.i32

# Set pool sizes for tiles (adjust as needed) - pre-allocate fields with large default sizes
DEFAULT_MAX_RAY_POOL = 16000000
DEFAULT_MAX_AMBIENT_REQUESTS = 2000000

ray_pool = Ray.field(shape=DEFAULT_MAX_RAY_POOL)
next_ray_pool = Ray.field(shape=DEFAULT_MAX_RAY_POOL)
ray_count        = ti.field(ti.i32, shape=())
next_ray_count   = ti.field(ti.i32, shape=())
ray_pool_overflow = ti.field(ti.i32, shape=())
ambient_pool_overflow = ti.field(ti.i32, shape=())

final_image      = None  # To be allocated later
num_pixels       = 0

ambient_requests = AmbientRequest.field(shape=DEFAULT_MAX_AMBIENT_REQUESTS)
ambient_req_count= ti.field(ti.i32, shape=())

#num_objects      = 10
#objects          = SceneObject.field(shape=num_objects)

samples_per_pixel = ti.field(ti.i32, shape=())
max_depth         = ti.field(ti.i32, shape=())
num_AO_samples    = ti.field(ti.i32, shape=())
max_AO_distance   = ti.field(ti.f32, shape=())

# Extra fields for scene and camera
eye           = ti.Vector.field(3, ti.f32, shape=())
light_pos     = ti.Vector.field(3, ti.f32, shape=())
ambient_color = ti.Vector.field(3, ti.f32, shape=())
light_color   = ti.Vector.field(3, ti.f32, shape=())
cam_origin            = ti.Vector.field(3, ti.f32, shape=())
cam_lower_left_corner = ti.Vector.field(3, ti.f32, shape=())
cam_horizontal        = ti.Vector.field(3, ti.f32, shape=())
cam_vertical          = ti.Vector.field(3, ti.f32, shape=())

# --- Utility Functions ---
@ti.func
def random_in_unit_sphere():
    p = 2.0 * ti.Vector([ti.random(), ti.random(), ti.random()]) - ti.Vector([1.0, 1.0, 1.0])
    while p.dot(p) >= 1.0:
        p = 2.0 * ti.Vector([ti.random(), ti.random(), ti.random()]) - ti.Vector([1.0, 1.0, 1.0])
        if p.dot(p) < 1.0:
            break
    return p

@ti.func
def random_hemisphere_direction(normal):
    u1 = ti.random()
    u2 = ti.random()
    phi = 2 * pi * u2
    cos_theta = ti.sqrt(u1)
    sin_theta = ti.sqrt(1.0 - u1)
    x = sin_theta * ti.cos(phi)
    y = sin_theta * ti.sin(phi)
    z = cos_theta
    local_dir = ti.Vector([x, y, z])
    tangent = ti.Vector([0.0, 0.0, 0.0])
    if ti.abs(normal[2]) < 0.999:
        tangent = normal.cross(ti.Vector([0.0, 0.0, 1.0]))
    else:
        tangent = normal.cross(ti.Vector([1.0, 0.0, 0.0]))
    tangent = tangent.normalized()
    bitangent = normal.cross(tangent)
    world_dir = local_dir[0] * tangent + local_dir[1] * bitangent + local_dir[2] * normal
    return world_dir.normalized()

@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()

@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = ti.min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(ti.abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * ti.pow((1 - cosine), 5)

# --- Scattering Functions ---
@ti.func
def scatter_lambertian(ray_direction, hit_point, normal, albedo):
    scatter_direction = random_hemisphere_direction(normal)
    return scatter_direction, albedo * (1.0/pi)

@ti.func
def scatter_metal(ray_direction, hit_point, normal, albedo, fuzz):
    reflected = reflect(ray_direction.normalized(), normal)
    scattered = reflected + fuzz * random_in_unit_sphere()
    scattered_dir = ti.Vector([0.0, 0.0, 0.0])
    attenuation = ti.Vector([0.0, 0.0, 0.0])
    if scattered.dot(normal) > 0:
        scattered_dir = scattered.normalized()
        attenuation = albedo
    return scattered_dir, attenuation

@ti.func
def scatter_dielectric(ray_direction, hit_point, normal, ref_idx):
    attenuation = ti.Vector([1.0, 1.0, 1.0])
    unit_direction = ray_direction.normalized()
    front_face = ray_direction.dot(normal) < 0
    etai_over_etat = 1.0 / ref_idx if front_face else ref_idx
    cos_theta = ti.min(normal.dot(-unit_direction), 1.0)
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
    cannot_refract = etai_over_etat * sin_theta > 1.0
    reflect_prob = reflectance(cos_theta, ref_idx)
    scattered = reflect(unit_direction, normal)
    if not cannot_refract and ti.random() >= reflect_prob:
        scattered = refract(unit_direction, normal, etai_over_etat)
    return scattered.normalized(), attenuation

@ti.func
def background_color(direction):
    unit_direction = direction.normalized()
    t_val = 0.5 * (unit_direction.y + 1.0)
    return (1.0 - t_val) * ti.Vector([1.0, 1.0, 1.0]) + t_val * ti.Vector([0.5, 0.7, 1.0])

# --- Ray Pool and Intersection Routines ---
@ti.func
def add_scattered_ray_to_pool(pixel_index, hit_point, depth, weight, scattered_direction):
    if next_ray_count[None] < max_ray_pool_param[None] - 1:  # Use parameter field
        idx = ti.atomic_add(next_ray_count[None], 1)
        next_ray_pool[idx] = Ray(pixel_index, hit_point + 0.001 * scattered_direction,
                                 scattered_direction.normalized(), depth - 1, weight, 0, -1)
    else:
        ray_pool_overflow[None] = 1

@ti.func
def trace_AO_ray(origin, direction, ambient_req_id):
    t = find_closest_intersection(origin, direction)[0]  # Only get the 't' value
    ti.atomic_add(ambient_requests[ambient_req_id].total, 1)
    if t < max_AO_distance[None]:
        ti.atomic_add(ambient_requests[ambient_req_id].occluded, 1)

@ti.func
def shadow_check(hit_point, normal):
    shadow_origin = hit_point + 0.001 * normal
    shadow_dir = (light_pos[None] - shadow_origin).normalized()
    t_shadow, _, _, _, _ = find_closest_intersection(shadow_origin, shadow_dir)
    dist_light = (light_pos[None] - shadow_origin).norm()
    return t_shadow < dist_light

@ti.func
def ray_sphere_intersection(ray_origin, ray_direction, center, radius, scale):
    t_result = float('inf')
    check_result = 0
    oc = ray_origin - center
    a = ray_direction.dot(ray_direction)
    b = 2.0 * oc.dot(ray_direction)
    c = oc.dot(oc) - radius * radius
    disc = b * b - 4 * a * c
    if disc > 0.0:
        t = (-b - ti.sqrt(disc)) / (2.0 * a)
        if t > 0.001:
            hit_point = ray_origin + t * ray_direction
            phi = ti.atan2(hit_point[2] - center[2], hit_point[0] - center[0])
            theta = ti.acos(ti.max(ti.min((hit_point[1] - center[1]) / radius, 1.0), -1.0))
            u = (phi + pi) / (2 * pi)
            v = theta / pi
            check = ((ti.floor(u * scale) + ti.floor(v * scale)) % 2) == 0
            t_result = t
            check_result = 1 if check else 0
    return t_result, check_result

@ti.func
def ray_plane_intersection(ray_origin, ray_direction, normal, offset, scale):
    t_result = float('inf')
    check_result = 0
    denom = normal.dot(ray_direction)
    if ti.abs(denom) > 0.0001:
        t = -(normal.dot(ray_origin) - offset) / denom
        if t > 0.001:
            hit_point = ray_origin + t * ray_direction
            x = ti.floor(hit_point[0] * scale)
            z = ti.floor(hit_point[2] * scale)
            t_result = t
            check_result = 1 if ((ti.floor(x) + ti.floor(z)) % 2) == 0 else 0
    return t_result, check_result

@ti.func
def find_closest_intersection(ray_origin, ray_direction):
    closest_t = float('inf')
    hit_object = -1
    is_checkboard = 0
    hit_point = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(num_objects)):
        obj = objects[i]
        t = float('inf')
        cb = 0
        if obj.type == 0:
            t, cb = ray_plane_intersection(ray_origin, ray_direction, obj.normal, obj.offset, obj.scale)
        elif obj.type == 1:
            t, cb = ray_sphere_intersection(ray_origin, ray_direction, obj.center, obj.radius, obj.scale)
        if t < closest_t:
            closest_t = t
            hit_object = i
            hit_point = ray_origin + t * ray_direction
            if obj.type == 0:
                normal = -obj.normal if obj.normal.dot(ray_direction) > 0 else obj.normal
            elif obj.type == 1:
                normal = (hit_point - obj.center) / obj.radius
            normal = normal.normalized()
            is_checkboard = cb
    return closest_t, hit_object, hit_point, normal, is_checkboard

@ti.func
def add_AO_rays_to_pool(pixel_index, hit_point, normal, weight, ambient_comp):
    if ambient_req_count[None] < max_ambient_requests_param[None] - 1:  # Use parameter field
        req_id = ti.atomic_add(ambient_req_count[None], 1)
        ambient_requests[req_id] = AmbientRequest(pixel_index, weight * ambient_comp, 0, 0)

        for s in range(num_AO_samples[None]):
            if next_ray_count[None] < max_ray_pool_param[None] - 1:  # Use parameter field
                ao_dir = random_hemisphere_direction(normal)
                idx = ti.atomic_add(next_ray_count[None], 1)
                next_ray_pool[idx] = Ray(pixel_index, hit_point + 0.001 * normal,
                                       ao_dir, 0, ti.Vector([1.0, 1.0, 1.0]), 1, req_id)
            else:
                ray_pool_overflow[None] = 1
    else:
        ambient_pool_overflow[None] = 1

@ti.func
def blinn_phong_shading(ray_direction, hit_point, normal, diffuse, specular):
    L = (light_pos[None] - hit_point).normalized()
    diff = ti.max(normal.dot(L), 0.0)
    V = (-ray_direction).normalized()
    H = (L + V).normalized()
    spec = ti.pow(ti.max(normal.dot(H), 0.0), 10)
    local_color = diffuse * diff + specular * spec
    shadow_origin = hit_point + 0.001 * normal
    t_shadow, hit_obj, _, _, _ = find_closest_intersection(shadow_origin, L)
    light_distance = (light_pos[None] - shadow_origin).norm()
    shadow_weight = 1.0
    if t_shadow < light_distance:
        if objects[hit_obj].material == 3:
            shadow_weight = 0.5
        else:
            shadow_weight = 0.0
    return local_color * shadow_weight

# --- Camera and Scene Setup ---
def setup_camera(image_width, image_height, cam_origin_arg, lookat_arg, vup_arg, theta_arg):
    cam_origin[None] = ti.Vector(cam_origin_arg)
    lookat = ti.Vector(lookat_arg)
    vup = ti.Vector(vup_arg).normalized()
    theta = theta_arg * (pi/180)  # Convert degrees to radians
    half_height = tan(theta / 2)
    aspect = image_width / image_height
    half_width = aspect * half_height
    w = (cam_origin[None] - lookat).normalized()
    u_cam = vup.cross(w).normalized()
    v_cam = w.cross(u_cam)
    cam_lower_left_corner[None] = cam_origin[None] - half_width * u_cam - half_height * v_cam - w
    cam_horizontal[None] = 2 * half_width * u_cam
    cam_vertical[None] = 2 * half_height * v_cam

@ti.kernel
def setup_tile(tile_x0: ti.i32, tile_y0: ti.i32, tile_w: ti.i32, tile_h: ti.i32,
               image_width: ti.i32, image_height: ti.i32):
    for y in range(tile_y0, tile_y0 + tile_h):
        for x in range(tile_x0, tile_x0 + tile_w):
            pixel = y * image_width + x
            for s in range(samples_per_pixel[None]):
                u = (x + ti.random()) / image_width
                v = 1.0 - (y + ti.random()) / image_height
                direction = (cam_lower_left_corner[None] + u * cam_horizontal[None] +
                             v * cam_vertical[None] - cam_origin[None]).normalized()
                idx = ti.atomic_add(ray_count[None], 1)
                ray_pool[idx] = Ray(pixel, cam_origin[None], direction, max_depth[None],
                                    ti.Vector([1.0, 1.0, 1.0]), 0, -1)

@ti.kernel
def process_rays():
    for i in range(ray_count[None]):
        if i >= max_ray_pool_param[None]:  # <-- Corrected line
            ti.static_print(f"Ray pool overflow at index {i}")
            continue
        ray = ray_pool[i]
        if ray.rt == 1:
            if ray.ambient_req_id >= max_ambient_requests_param[None]:  # <-- Corrected line
                ti.static_print(f"Ambient request overflow: {ray.ambient_req_id}")
                continue
            trace_AO_ray(ray.origin, ray.direction, ray.ambient_req_id)
            continue

        if ray.depth <= 0:
            continue
        if ray.depth < max_depth[None]:
            if ti.random() > RR_prob[None]:  # Fixed here
                continue
            else:
                ray.weight /= RR_prob[None]  # Fixed here
        t, hit_obj, hit_point, normal, is_check = find_closest_intersection(ray.origin, ray.direction)
        if hit_obj < 0:
            final_image[ray.pixel_index] += ray.weight * background_color(ray.direction)
            continue
        obj = objects[hit_obj]
        if obj.material == 0:
            final_image[ray.pixel_index] += ray.weight * obj.diffuse
            continue
        if ENABLE_DIRECT_LIGHTING[None]:  # Fixed here
            direct_color = blinn_phong_shading(ray.direction, hit_point, normal, obj.diffuse, obj.specular)
            final_image[ray.pixel_index] += ray.weight * direct_color
        scattered = ti.Vector([0.0, 0.0, 0.0])
        attenuation = ti.Vector([0.0, 0.0, 0.0])
        if obj.material == 1:
            scattered, attenuation = scatter_lambertian(ray.direction, hit_point, normal, obj.diffuse)
        elif obj.material == 2:
            scattered, attenuation = scatter_metal(ray.direction, hit_point, normal, obj.specular, 0.0)
        elif obj.material == 3:
            scattered, attenuation = scatter_dielectric(ray.direction, hit_point, normal, obj.ior)
        elif obj.material == 4:
            scattered, attenuation = scatter_metal(ray.direction, hit_point, normal, obj.specular, obj.fuzz)
        new_weight = ray.weight * attenuation
        if scattered.norm() > 0 and ray.depth > 1:
            add_scattered_ray_to_pool(ray.pixel_index, hit_point, ray.depth, new_weight, scattered)
        if ENABLE_AO[None]:  # Fixed here
            add_AO_rays_to_pool(ray.pixel_index, hit_point, normal, ray.weight, obj.diffuse)
                        
@ti.kernel
def update_ray_pool(new_count: ti.i32):
    for i in range(new_count):
        ray_pool[i] = next_ray_pool[i]
    ray_count[None] = new_count

@ti.kernel
def resolve_ambient():
    for i in range(ambient_req_count[None]):
        req = ambient_requests[i]
        idx = req.pixel_index
        factor = 1.0
        if req.total > 0:
            factor = 1.0 - (req.occluded / req.total)
        final_image[idx] += req.ambient_contrib * factor

def setup_scene(image_width, image_height, enable_ao, enable_direct, rr_prob,
                samples_per_pixel_val, max_depth_val, num_AO_samples_val, max_AO_distance_val,
                light_pos_arg, ambient_color_arg, light_color_arg, 
                cam_origin_arg, lookat_arg, vup_arg, theta_arg,
                objects_arg):  # Add this parameter
    global num_objects, objects
    
    # Clear existing objects
    num_objects = len(objects_arg)
    objects = SceneObject.field(shape=num_objects)

    for i, obj_config in enumerate(objects_arg):
        objects[i] = SceneObject(
            type=obj_config["type"],
            center=ti.Vector(obj_config.get("center", [0,0,0])),
            radius=obj_config.get("radius", 1.0),
            normal=ti.Vector(obj_config.get("normal", [0,0,0])),
            offset=obj_config.get("offset", 0.0),
            diffuse=ti.Vector(obj_config["diffuse"]),
            specular=ti.Vector(obj_config.get("specular", [0,0,0])),
            ior=obj_config.get("ior", 1.0),
            do_checkboard=obj_config.get("do_checkboard", 0),
            scale=obj_config.get("scale", 1),
            fuzz=obj_config.get("fuzz", 0.0),
            material=obj_config["material"]
        )
    
    global num_pixels, final_image
    ENABLE_AO[None] = enable_ao
    ENABLE_DIRECT_LIGHTING[None] = enable_direct
    RR_prob[None] = rr_prob
    num_pixels = image_width * image_height
    final_image = ti.Vector.field(3, ti.f32, shape=(num_pixels,))
    final_image.fill(0)
    samples_per_pixel[None] = samples_per_pixel_val
    max_depth[None] = max_depth_val
    num_AO_samples[None] = num_AO_samples_val
    max_AO_distance[None] = max_AO_distance_val
    light_pos[None] = ti.Vector(light_pos_arg)
    ambient_color[None] = ti.Vector(ambient_color_arg)  # Use parameter
    light_color[None] = ti.Vector(light_color_arg)      # Use parameter
    setup_camera(image_width, image_height, cam_origin_arg, lookat_arg, vup_arg, theta_arg)

    # Define scene objects.
    '''objects[0] = SceneObject(1, ti.Vector([0.0, 5.4, -1.0]), 3.0,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([10.0, 10.0, 10.0]), ti.Vector([0.0, 0.0, 0.0]),
                             1.0, 0, 1, 0.0, 0)
    objects[1] = SceneObject(1, ti.Vector([0.0, -100.5, -1.0]), 100.0,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([0.8, 0.8, 0.8]), ti.Vector([0.0, 0.0, 0.0]),
                             1.0, 0, 1, 0.0, 1)
    objects[2] = SceneObject(1, ti.Vector([0.0, 102.5, -1.0]), 100.0,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([0.8, 0.8, 0.8]), ti.Vector([0.0, 0.0, 0.0]),
                             1.0, 0, 1, 0.0, 1)
    objects[3] = SceneObject(1, ti.Vector([0.0, 1.0, 101.0]), 100.0,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([0.8, 0.8, 0.8]), ti.Vector([0.0, 0.0, 0.0]),
                             1.0, 0, 1, 0.0, 1)
    objects[4] = SceneObject(1, ti.Vector([-101.5, 0.0, -1.0]), 100.0,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([0.6, 0.0, 0.0]), ti.Vector([0.0, 0.0, 0.0]),
                             1.0, 0, 1, 0.0, 1)
    objects[5] = SceneObject(1, ti.Vector([101.5, 0.0, -1.0]), 100.0,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([0.0, 0.6, 0.0]), ti.Vector([0.0, 0.0, 0.0]),
                             1.0, 0, 1, 0.0, 1)
    objects[6] = SceneObject(1, ti.Vector([0.0, -0.2, -1.5]), 0.3,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([0.8, 0.3, 0.3]), ti.Vector([0.0, 0.0, 0.0]),
                             1.0, 0, 1, 0.0, 1)
    objects[7] = SceneObject(1, ti.Vector([-0.8, 0.2, -1.0]), 0.7,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.6, 0.8, 0.8]),
                             1.0, 0, 1, 0.0, 2)
    objects[8] = SceneObject(1, ti.Vector([0.7, 0.0, -0.5]), 0.5,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([1.0, 1.0, 1.0]), ti.Vector([0.0, 0.0, 0.0]),
                             1.5, 0, 1, 0.0, 3)
    objects[9] = SceneObject(1, ti.Vector([0.6, -0.3, -2.0]), 0.2,
                             ti.Vector([0.0, 0.0, 0.0]), 0.0,
                             ti.Vector([0.0, 0.0, 0.0]), ti.Vector([0.8, 0.6, 0.2]),
                             1.0, 0, 1, 0.1, 4)'''

def render_tiled(image_width, image_height, tile_w, tile_h, enable_ao, enable_direct, rr_prob,
                 max_ray_pool, max_ambient_requests, 
                 samples_per_pixel_val=256, max_depth_val=8, num_AO_samples_val=32, max_AO_distance_val=2.0,
                 light_pos_arg=None, ambient_color_arg=None, light_color_arg=None, 
                 cam_origin_arg=None, lookat_arg=None, vup_arg=None, theta_arg=60.0,
                 objects_arg=None):  # Add this parameter
    # Set parameter values into Taichi fields
    max_ray_pool_param[None] = max_ray_pool
    max_ambient_requests_param[None] = max_ambient_requests
    # Pass new parameters to setup_scene
    setup_scene(image_width, image_height, enable_ao, enable_direct, rr_prob,
            samples_per_pixel_val, max_depth_val, num_AO_samples_val, max_AO_distance_val,
            light_pos_arg, ambient_color_arg, light_color_arg, 
            cam_origin_arg, lookat_arg, vup_arg, theta_arg,
            objects_arg)
    start_time = time.time()
    for tile_y in range(0, image_height, tile_h):
        for tile_x in range(0, image_width, tile_w):
            ray_pool_overflow[None] = 0
            ambient_pool_overflow[None] = 0

            cur_tile_w = min(tile_w, image_width - tile_x)
            cur_tile_h = min(tile_h, image_height - tile_y)

            ray_count[None] = 0
            next_ray_count[None] = 0
            ambient_req_count[None] = 0

            ray_pool.fill(0)
            next_ray_pool.fill(0)
            ambient_requests.fill(0)

            setup_tile(tile_x, tile_y, cur_tile_w, cur_tile_h, image_width, image_height)
            total_tile_rays = 0

            safety_counter = 0
            max_iterations = 100

            while ray_count[None] > 0 and safety_counter < max_iterations:
                safety_counter += 1
                total_tile_rays += ray_count[None]
                next_ray_count[None] = 0
                process_rays()
                update_ray_pool(next_ray_count[None])

            ti.sync()
            if ray_pool_overflow[None]:
                print(f"Warning: Ray pool capacity reached in tile ({tile_x},{tile_y})")
            if ambient_pool_overflow[None]:
                print(f"Warning: Ambient request pool capacity reached in tile ({tile_x},{tile_y})")
            resolve_ambient()
            end_time = time.time()
            print(f"Tile ({tile_x},{tile_y}) processed in {end_time - start_time:.2f} s with {total_tile_rays} rays.")
    for i in range(num_pixels):
        final_image[i] = final_image[i] / samples_per_pixel[None]
    return final_image.to_numpy()

def render(image_width=640, image_height=360, tile_w=16, tile_h=16, display=True,
           ENABLE_AO=True, ENABLE_DIRECT_LIGHTING=True, RR_prob=0.8,
           max_ray_pool=8000000, max_ambient_requests=1000000,
           samples_per_pixel=256, max_depth=8, num_AO_samples=32, max_AO_distance=2.0,
           light_pos=None, ambient_color=None, light_color=None,
           cam_origin=None, lookat=None, vup=None, theta=None,
           objects=None, config_file='raychi_config.json'):

    """
    Render an image using the RayChi engine.

    Args:
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.
        tile_w (int): Tile width.
        tile_h (int): Tile height.
        display (bool): If True, the image is displayed using matplotlib.
        ENABLE_AO (bool): Enable ambient occlusion.
        ENABLE_DIRECT_LIGHTING (bool): Enable direct lighting.
        RR_prob (float): Russian Roulette probability (0.8 recommended).
        max_ray_pool (int): Logical max ray pool size.
        max_ambient_requests (int): Logical max ambient requests.
        samples_per_pixel (int): Number of samples per pixel.
        max_depth (int): Maximum ray bounce depth.
        num_AO_samples (int): Number of ambient occlusion samples.
        max_AO_distance (float): Maximum AO ray distance.
        light_pos (list/tuple): 3-element list/tuple for light position.
        objects (list): List of scene object configurations
        config_file (str): Path to configuration file
    """
    # Handle configuration parameters
    config = None
    required_params = {
        'light_pos': light_pos,
        'ambient_color': ambient_color,
        'light_color': light_color,
        'cam_origin': cam_origin,
        'lookat': lookat,
        'vup': vup,
        'theta': theta,
        'objects': objects
    }

    # Load config if any required parameter is missing
    if any(v is None for v in required_params.values()):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            # Set missing params from config
            for param in required_params:
                if required_params[param] is None:
                    if param not in config:
                        if param == 'objects':
                            # Objects are mandatory - no default
                            print("Error: 'objects' configuration missing")
                            sys.exit(1)
                        print(f"Error: '{param}' missing in config file.")
                        sys.exit(1)
                    required_params[param] = config[param]
        except FileNotFoundError:
            missing = [k for k, v in required_params.items() if v is None]
            print(f"Error: Config file '{config_file}' not found and missing: {', '.join(missing)}.")
            sys.exit(1)

    # Unpack and validate parameters
    light_pos = required_params['light_pos']
    ambient_color = required_params['ambient_color']
    light_color = required_params['light_color']
    cam_origin = required_params['cam_origin']
    lookat = required_params['lookat']
    vup = required_params['vup']
    theta = required_params['theta']
    objects_config = required_params['objects']

    # Validate basic parameter types
    for name, val in [('light_pos', light_pos),
                     ('ambient_color', ambient_color),
                     ('light_color', light_color),
                     ('cam_origin', cam_origin),
                     ('lookat', lookat),
                     ('vup', vup)]:
        if not (isinstance(val, (list, tuple)) and len(val) == 3):
            print(f"Error: {name} must be a 3-element list/tuple.")
            sys.exit(1)

    if len(objects_config) < 1:
        print("Error: Scene must contain at least 1 object")
        sys.exit(1)
    
    if not isinstance(theta, (float, int)) or theta <= 0:
        print("Error: theta must be a positive number.")
        sys.exit(1)

    # Validate objects configuration
    if not isinstance(objects_config, list):
        print("Error: Objects configuration must be a list")
        sys.exit(1)
        
    for i, obj in enumerate(objects_config):
        if 'type' not in obj:
            print(f"Error: Object {i} missing 'type' specification")
            sys.exit(1)
            
        obj_type = obj['type']
        if obj_type == 1:  # Sphere
            required = ['center', 'radius', 'diffuse', 'material']
        elif obj_type == 0:  # Plane
            required = ['normal', 'offset', 'diffuse', 'material']
        else:
            print(f"Error: Unknown object type {obj_type} in object {i}")
            sys.exit(1)
            
        for field in required:
            if field not in obj:
                print(f"Error: Object {i} missing required field '{field}'")
                sys.exit(1)

    # Pass validated parameters to renderer
    img = render_tiled(image_width, image_height, tile_w, tile_h, 
                      ENABLE_AO, ENABLE_DIRECT_LIGHTING, RR_prob,
                      max_ray_pool, max_ambient_requests,
                      samples_per_pixel, max_depth, num_AO_samples, max_AO_distance,
                      light_pos_arg=light_pos,
                      ambient_color_arg=ambient_color,
                      light_color_arg=light_color,
                      cam_origin_arg=cam_origin,
                      lookat_arg=lookat,
                      vup_arg=vup,
                      theta_arg=theta,
                      objects_arg=objects_config)

    # Post-processing and display
    gamma = 2.2
    img = np.power(img, 1.0/gamma)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    if display:
        plt.figure(figsize=(12, 6))
        plt.imshow(img.reshape(image_height, image_width, 3))
        plt.axis('off')
        plt.show()
        
    return img