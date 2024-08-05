bl_info = {
    "name": "Low Poly Remeshing",
    "author": "White Water",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > N",
    "description": "Turn a mesh with high number of triangles to a low one",
    "warning": "",
    "doc_url": "",
    "category": "Mesh",
}


import bpy
import math
import bmesh
import mc33
import sys
import time
import mathutils
from bpy.props import BoolProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup

#-------------------------------------- Hype Parameters -------------------------------------#
class HyperParameters : 
    d = 0
    l = 0
    np = 20
    voxel_length = 1
    r = 0.125 # used in alignment
    l_0 = 8
    theta_0 = 120 # in degree
    epsilon = 0.0001
    d_iso_mesh = None

def create_point_at_position(position: tuple) :
    bpy.ops.mesh.dupli_extrude_cursor(rotate_source=True)
    bpy.ops.transform.translate(value=position, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
    return

def create_new_mesh(verts, faces, mesh_name) :
    # 创建 mesh 对象
    d_iso_mesh = bpy.data.meshes.new("new_d_iso_mesh")
    d_iso_mesh.from_pydata(verts, [], faces)
    d_iso_mesh.update()

    # 创建 object 对象
    obj = bpy.data.objects.new(mesh_name, d_iso_mesh)

    # 将 object 放进 某个 Collection 中
    col =  bpy.data.collections["Collection"]
    col.objects.link(obj)
    return d_iso_mesh

def create_test_mesh() :
    # 首先定义顶点的信息
    verts = [
        (1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0)
    ]

    # 定义面索引，这里用的不是三角形，而是多边形，对应正方体的 6 个面
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (4, 0, 3, 7)
    ]  

    # 创建 mesh 对象
    mesh_data = bpy.data.meshes.new("cube_mesh_data")
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    # 创建 object 对象
    obj = bpy.data.objects.new("My_Object", mesh_data)

    # 将 object 放进 某个 Collection 中
    col =  bpy.data.collections["Collection"]
    col.objects.link(obj)
    return

def NullFunc(arg1: float, arg2: float, arg3: float)->float :
    return 1

def print_all_grid_values(grid) :
    for i in range(0, grid.get_N_i(0) + 1) :
        for j in range(0, grid.get_N_i(1) + 1) :
            for k in range(0, grid.get_N_i(2) + 1) :
                print(grid.get_grid_value(i, j, k))

def print_all_grid_values_greater_than_val(grid, val) :
    total = 0
    for i in range(0, grid.get_N_i(0) + 1) :
        for j in range(0, grid.get_N_i(1) + 1) :
            for k in range(0, grid.get_N_i(2) + 1) :
                if grid.get_grid_value(i, j, k) > val :
                    total += 1
                    print(grid.get_grid_value(i, j, k))
    print('total > val: ', total)

def print_all_grid_values_less_than_val(grid, val) :
    total = 0
    for i in range(0, grid.get_N_i(0) + 1) :
        for j in range(0, grid.get_N_i(1) + 1) :
            for k in range(0, grid.get_N_i(2) + 1) :
                if grid.get_grid_value(i, j, k) < val :
                    total += 1
                    print(grid.get_grid_value(i, j, k))
    print('total < val: ', total)

def get_vector_length(x, y, z) :
    return math.sqrt(x * x + y * y + z * z)

def dot_product(A:mathutils.Vector, B:mathutils.Vector)->float :
    return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]

def cross_product(A:mathutils.Vector, B:mathutils.Vector)->mathutils.Vector :
    return mathutils.Vector((A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]))

# 默认了体素是正方体，所以步长step只有一个
def calculate_grid_iso(x_start, y_start, z_start, x_end, y_end, z_end, step, grid, bmesh) :
    x = x_start
    y = y_start
    z = z_start

    total_set = 0
    for k in range(0, grid.get_N_i(2) + 1) :
        y = y_start;
        for j in range(0, grid.get_N_i(1) + 1) :
            x = x_start;
            for i in range(0, grid.get_N_i(0) + 1) :
                grid.set_grid_value(i, j, k, get_UDF_isovalue(x, y, z, bmesh))
                total_set += 1
                x += step;
            y += step;
        z += step;

def check_grid_iso_values(x_start, y_start, z_start, step, grid) :
    x = x_start
    y = y_start
    z = z_start

    print('check starts.')
    for k in range(0, grid.get_N_i(2) + 1) :
        y = y_start;
        for j in range(0, grid.get_N_i(1) + 1) :
            x = x_start;
            for i in range(0, grid.get_N_i(0) + 1) :
                iso_val = grid.get_grid_value_direct(i, j, k, False)
                if iso_val < HyperParameters.d :
                    print(iso_val)
                x += step;
            y += step;
        z += step;
    print('check ends.')

def create_d_iso_cube(x_start, y_start, z_start, step, grid) :
    x = x_start
    y = y_start
    z = z_start
    print('total vertices in cube: ', (grid.get_N_i(0) + 1) * (grid.get_N_i(1) + 1) * (grid.get_N_i(2) + 1))

    iso_cube_verts = []
    iso_values = []
    for k in range(0, grid.get_N_i(2) + 1) :
        y = y_start;
        for j in range(0, grid.get_N_i(1) + 1) :
            x = x_start;
            for i in range(0, grid.get_N_i(0) + 1) :
                iso_val = grid.get_grid_value_direct(i, j, k, False)
                iso_values.append(iso_val)

                iso_cube_verts.append((x, y, z))
                x += step;
            y += step;
        z += step;

    iso_mesh = create_new_mesh(iso_cube_verts, [], "iso_mesh")
    i = 0
    sum = 0
    for v in iso_mesh.vertices :
        if iso_values[i] > HyperParameters.d :
            sum += 1
            v.select = False
        i += 1

    print('num greater than d: ', sum)

def in_triangle(A:mathutils.Vector, B:mathutils.Vector, C:mathutils.Vector, P:mathutils.Vector)->bool :
    AB = B - A
    AC = C - A
    PA = A - P
    vx = mathutils.Vector((AB[0], AC[0], PA[0]))
    vy = mathutils.Vector((AB[1], AC[1], PA[1]))
    u = mathutils.Vector((0., 0., 0.))
    w = cross_product(vx, vy)
    if w[2] < 0 :
        u = mathutils.Vector((-w[0], -w[1], -w[2]))
    else :
        u = w

    if (u[0] < 0) or (u[1] < 0) or (u[0] + u[1] > u[2]) :
        return False
    return True

def min_dist_to_triangle(x, y, z, face) :
    n = face.normal
    o = mathutils.Vector((x, y, z))
    d = mathutils.Vector((-n[0], -n[1], -n[2]))     # o + t * d  才是射线，这里法线取反比较方便                   
    p_prime = face.verts[0].co         # p_prime实际上就是三角形内一点，这里直接取第一个点
    t = (o - p_prime).dot(n)
    p = o + t * d

    if in_triangle(face.verts[0].co, face.verts[1].co, face.verts[2].co, p) :
        return (t * d).length, d, n, (t * d).length
    
    min_dist = sys.float_info.max
    temp_min_dist = 0
    _d = mathutils.Vector((0., 0., 0.))
    middle_point = mathutils.Vector((0., 0., 0.))
    for v in face.verts :
        middle_point += v.co
        temp_min_dist = (o - v.co).length
        if temp_min_dist < min_dist :
            min_dist = temp_min_dist
            _d = (v.co - o)

    middle_point /= 3

    return min_dist, _d, n, (o - middle_point).length

def get_UDF_isovalue(x, y, z, bmesh) :
    min_dist = sys.float_info.max
    min_dist_to_tri = sys.float_info.max
    min_d = mathutils.Vector((0., 0., 0.))
    min_n = mathutils.Vector((0., 0., 0.))
    
    temp_min_dist = 0.
    temp_min_dist_to_tri = 0.       # 还需要记录到三角形的距离，否则会出现某个三角形的点离网格点最近，但该三角形点不止存在于一个三角形内。该距离是网格点到三角形中心的距离
    # 上述方法还存在缺陷，假设网格点离某三角形点最近，该三角形点由两个三角形共享，一个三角形太小，另一个三角形太大，结果网格点到达较小三角形的中点就更近，尽管因该离较大三角形较近
    # 解决思路看github的图
    n = mathutils.Vector((0., 0., 0.))
    d = mathutils.Vector((0., 0., 0.))
    
    for f in bmesh.faces :
        temp_min_dist, d, n, temp_min_dist_to_tri = min_dist_to_triangle(x, y, z, f)    # 除了返回最小距离，还返回方向向量d，用于计算符号
        if temp_min_dist <= min_dist :
            if temp_min_dist_to_tri < min_dist_to_tri :
                min_dist = temp_min_dist
                min_dist_to_tri = temp_min_dist_to_tri
                min_d = d
                min_n = n
    
    # 确定符号
    # if min_d.dot(min_n) > 0 :
    #     return -min_dist
    # else :
    #     return min_dist

    return min_dist

#-------------------------------------- Functionalities -------------------------------------#
def discretize(self, context) :
    # init
    context.scene.low_poly_meshing_target.location = (0, 0, 0)
    context.scene.low_poly_meshing_target.rotation_euler = (0, 0, 0)
    context.scene.low_poly_meshing_target.scale = (1, 1, 1)
#    space_data = bpy.context.space_data
#    view_matrix = space_data.region_3d.view_matrix
#    perspective_matrix = space_data.region_3d.perspective_matrix
#    print(view_matrix)
#    print(perspective_matrix)

    # start discretizing
    mesh = context.scene.low_poly_meshing_target.data
    my_bmesh = bmesh.new()
    my_bmesh.from_mesh(mesh)

    # for f in mesh.polygons:
    #     # 获取每个面的法向量
    #     print(f.index, f.normal)
    
    # find bounding-box
    min_x = float("inf")
    min_y = float("inf")
    min_z = float("inf")
    
    max_x = float("-inf")
    max_y = float("-inf")
    max_z = float("-inf")
    
    for v in my_bmesh.verts:
        if (v.co.x < min_x) :
            min_x = v.co.x
        if (v.co.y < min_y) :
            min_y = v.co.y
        if (v.co.z < min_z) :
            min_z = v.co.z
            
        if (v.co.x > max_x) :
            max_x = v.co.x
        if (v.co.y > max_y) :
            max_y = v.co.y
        if (v.co.z > max_z) :
            max_z = v.co.z

    length_bbox = (max_x - min_x, max_y - min_y, max_z - min_z)
    location_bbox = ((max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2)

    total_length = length_bbox[0] + length_bbox[1] + length_bbox[2]
    temp_length_bbox = (length_bbox[0] / total_length, length_bbox[1] / total_length, length_bbox[2] / total_length)
    # Model is facing y-axis by default, so just project diagonal length of bbox onto plane perpendicular to y-axis(which is XOZ plane), we get l
    HyperParameters.l = math.sqrt(length_bbox[0] * length_bbox[0] + length_bbox[2] * length_bbox[2])
    HyperParameters.d = HyperParameters.l / HyperParameters.np
    HyperParameters.voxel_length = HyperParameters.d / math.sqrt(3)

    grid_origin = (min_x, min_y, min_z)
    grid_origin_opposite = (max_x, max_y, max_z)
    
    if (HyperParameters.l <= 0) :
        self.report({'ERROR'}, 'Hype Parameter l is less equal to 0.')
        my_bmesh.free()
        return

    # Timer
    start_time = time.time()

    mc = mc33.MC33()
    _surface = mc33.surface()
    grid = mc33.grid3d()
    two_d = 2 * HyperParameters.d
    grid.generate_grid(min_x - two_d, min_y - two_d, min_z - two_d, max_x + two_d, max_y + two_d, max_z + two_d, HyperParameters.voxel_length, HyperParameters.voxel_length, HyperParameters.voxel_length, False)
    calculate_grid_iso(min_x - two_d, min_y - two_d, min_z - two_d, max_x + two_d, max_y + two_d, max_z + two_d, HyperParameters.voxel_length, grid, my_bmesh)

    create_d_iso_cube(min_x - two_d, min_y - two_d, min_z - two_d, HyperParameters.voxel_length, grid)

    mc.set_grid3d(grid)
    # mc.calculate_isosurface(_surface, HyperParameters.d)
    print('Done calculating.')
    print('Triangle num: ', _surface.get_num_triangles())
    print('Vertex num: ', _surface.get_num_vertices())

    d_iso_verts = []
    d_iso_faces = []
    
    for tri_index in range(0, _surface.get_num_triangles()) :
        verts_index_in_face_vec = mathutils.Vector((0, 0, 0))
        for i in range(0, 3) :
            vert_index = _surface.getTriangle_vertex_i(tri_index, i)
            vert_co = mathutils.Vector((0., 0., 0.))
            vert_normal = mathutils.Vector((0., 0., 0.))
            for j in range(0, 3) :
                vert_co[j] = _surface.getVertex_co_i(vert_index, j)
                vert_normal[j] = _surface.getVertex_co_i(vert_index, j)
            vert = (vert_co[0], vert_co[1], vert_co[2])
            verts_index_in_face_vec[i] = len(d_iso_verts)  # 要放入d_iso_faces中的下标
            d_iso_verts.append(vert)
        verts_index_in_face = (int(verts_index_in_face_vec[0]), int(verts_index_in_face_vec[1]), int(verts_index_in_face_vec[2]))
        d_iso_faces.append(verts_index_in_face)

    print('len of verts: ', len(d_iso_verts))
    print('len of faces: ', len(d_iso_faces))

    # create_new_mesh(d_iso_verts, d_iso_faces, "d_iso")

    end_time = time.time()
    print('Time used: ', end_time - start_time)

    my_bmesh.to_mesh(mesh)
    my_bmesh.free()
    return

#---------------------------------------- Properties ----------------------------------------#
# Work as indicators in UI, not actually control the process
class PropsInUI(PropertyGroup):
    def check_finished_discretization(self, context) :
        if (LowPolyRemeshingProps.discretization_finished and self.discretization_finished) :
            return
        
        if ((not LowPolyRemeshingProps.discretization_finished) and (not self.discretization_finished)) :
            return
        
        if (LowPolyRemeshingProps.discretization_finished) : 
            self.discretization_finished = True
            
        if (not LowPolyRemeshingProps.discretization_finished) : 
            self.discretization_finished = False
        
        return
        
    discretization_finished : BoolProperty(
        name = "Enable or Disable",
        description = "Is discretization finished",
        update = check_finished_discretization,
        default = False
        )
        
    
    def check_finished_d_iso(self, context) :
        if (LowPolyRemeshingProps.d_iso_extraction_finished and self.d_iso_extraction_finished) :
            return
        
        if ((not LowPolyRemeshingProps.d_iso_extraction_finished) and (not self.d_iso_extraction_finished)) :
            return
        
        if (LowPolyRemeshingProps.d_iso_extraction_finished) : 
            self.d_iso_extraction_finished = True
            
        if (not LowPolyRemeshingProps.d_iso_extraction_finished) : 
            self.d_iso_extraction_finished = False
        
        return
    
    d_iso_extraction_finished : BoolProperty(
        name = "Enable or Disable",
        description = "Is d-isosurface extraction finished",
        update = check_finished_d_iso,
        default = False
        )
        
    
    def check_finished_sharp_recovery(self, context) :
        if (LowPolyRemeshingProps.sharp_features_recovery_finished and self.sharp_features_recovery_finished) :
            return
        
        if ((not LowPolyRemeshingProps.sharp_features_recovery_finished) and (not self.sharp_features_recovery_finished)) :
            return
        
        if (LowPolyRemeshingProps.sharp_features_recovery_finished) : 
            self.sharp_features_recovery_finished = True
            
        if (not LowPolyRemeshingProps.sharp_features_recovery_finished) : 
            self.sharp_features_recovery_finished = False
        
        return
        
    sharp_features_recovery_finished : BoolProperty(
        name = "Enable or Disable",
        description = "Is sharp features recovery finished",
        update = check_finished_sharp_recovery,
        default = False
        )
        
    test_bool : BoolProperty(
        name = "Enable or Disable",
        description = "Is sharp features recovery finished",
        default = False
        )

# Actual properties to control process
class LowPolyRemeshingProps: 
    discretization_finished = False
    d_iso_extraction_finished = False
    sharp_features_recovery_finished = False

#------------------------------------------ Buttons -----------------------------------------#
class DiscretizationOperator(bpy.types.Operator):
    """Start discretization of mesh"""
    bl_idname = "view3d.low_poly_remeshing_discretize"
    bl_label = "Discretize"

    def execute(self, context):
        ui_props = context.object.ui_props
        
        # guard
        if (LowPolyRemeshingProps.discretization_finished) :
            self.report({'ERROR'}, 'This step is already done.')
            ui_props.discretization_finished = True
            return {'CANCELLED'}
        
        # functionality
        if (context.scene.low_poly_meshing_target is None) :
            self.report({'ERROR'}, 'No target selected.')
            return {'CANCELLED'}

        discretize(self, context)
        print('new d is: ' + str(HyperParameters.d))

        # mesh = context.scene.low_poly_meshing_target.data
        # my_bmesh = bmesh.new()
        # my_bmesh.from_mesh(mesh)

        ui_props.discretization_finished = True
        LowPolyRemeshingProps.discretization_finished = True
        return {'FINISHED'}
    
    
    
class D_IsoSurfaceExtractionOperator(bpy.types.Operator):
    """Start d-isosurface extraction"""
    bl_idname = "view3d.low_poly_remeshing_d_iso_extract"
    bl_label = "D_IsoSurface Extract"

    def execute(self, context):
        ui_props = context.object.ui_props
        
        # guard
        if (LowPolyRemeshingProps.d_iso_extraction_finished) :
            self.report({'ERROR'}, 'This step is already done.')
            return {'CANCELLED'}
        
        if (not LowPolyRemeshingProps.discretization_finished):
            self.report({'ERROR'}, 'Please finish all previous steps in order.')
            return {'CANCELLED'}
        
        # functionality
        print(math.sqrt(2))
        LowPolyRemeshingProps.d_iso_extraction_finished = True
        ui_props.d_iso_extraction_finished = True
        return {'FINISHED'}
    
    
    
class SharpFeaturesRecoveryOperator(bpy.types.Operator):
    """Start recovering sharp features"""
    bl_idname = "view3d.low_poly_remeshing_sharp_features_recovery"
    bl_label = "Recover Sharp Features"

    def execute(self, context):
        ui_props = context.object.ui_props
        
        # guard
        if (LowPolyRemeshingProps.sharp_features_recovery_finished) :
            self.report({'ERROR'}, 'This step is already done.')
            return {'CANCELLED'}
        
        if (not LowPolyRemeshingProps.discretization_finished or not LowPolyRemeshingProps.d_iso_extraction_finished):
            self.report({'ERROR'}, 'Please finish all previous steps in order.')
            return {'CANCELLED'}
        
        # functionality
        LowPolyRemeshingProps.sharp_features_recovery_finished = True
        ui_props.sharp_features_recovery_finished = True
        return {'FINISHED'}

class ResetOperator(bpy.types.Operator):
    """Reset all flags of previous steps"""
    bl_idname = "view3d.low_poly_remeshing_reset"
    bl_label = "Reset"
    
    def execute(self, context):
        print('reset')
        ui_props = context.object.ui_props
        
        LowPolyRemeshingProps.discretization_finished = False
        ui_props.discretization_finished = False
        
        LowPolyRemeshingProps.d_iso_extraction_finished = False
        ui_props.d_iso_extraction_finished = False
        
        LowPolyRemeshingProps.sharp_features_recovery_finished = False
        ui_props.sharp_features_recovery_finished = False
        return {'FINISHED'}
    
#------------------------------------------- Panel ------------------------------------------#
class LowPolyMeshingPanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Low Poly Remeshing"
    bl_idname = "VIEW_3D_PT_LowPolyMeshingPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Low Poly Remeshing"


    def draw(self, context):
        layout = self.layout
        scene = context.scene
        object = context.object
        # props = object.low_poly_remeshing_props
        ui_props = object.ui_props
        
        # display the properties and buttons    
        layout.prop(scene, "low_poly_meshing_target", text="Target")
        
        row = layout.row()
        row.prop(ui_props, "discretization_finished", text="")
        row.operator(DiscretizationOperator.bl_idname, icon="SPLIT_HORIZONTAL")
        
        row = layout.row()
        row.prop(ui_props, "d_iso_extraction_finished", text="")
        row.operator(D_IsoSurfaceExtractionOperator.bl_idname, icon="EVENT_D")
        
        row = layout.row()
        row.prop(ui_props, "sharp_features_recovery_finished", text="")
        row.operator(SharpFeaturesRecoveryOperator.bl_idname, icon="EVENT_S")
        
        row = layout.row()
        row.operator(ResetOperator.bl_idname, icon="EVENT_R")
        

#---------------------------------------- Registration --------------------------------------#
from bpy.utils import register_class, unregister_class

all_classes={
    DiscretizationOperator,
    D_IsoSurfaceExtractionOperator,
    SharpFeaturesRecoveryOperator,
    ResetOperator,
    LowPolyMeshingPanel,
    #LowPolyRemeshingProps,
    PropsInUI
}


def register():
    for cls in all_classes:
        register_class(cls)

    #bpy.types.Object.low_poly_remeshing_props = bpy.props.PointerProperty(type = LowPolyRemeshingProps)
    bpy.types.Object.ui_props = bpy.props.PointerProperty(type = PropsInUI)
        
    bpy.types.Scene.low_poly_meshing_target = PointerProperty(
        type = bpy.types.Object,
        name = "low_poly_meshing_target"
    )
    
    test_bool = BoolProperty(
        name="Enable or Disable",
        description="Is discretization finished",
        is_readonly = False,
        is_hidden = True,
        default = False
    )
    

def unregister():
    for cls in all_classes:
        unregister_class(cls)
    
    del bpy.types.Scene.low_poly_meshing_target


if __name__ == "__main__":
    register()
