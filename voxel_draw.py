bl_info = {
    "name": "VoxelDraw",
    "author": "Sreeraj R",
    "version": (1, 0),
    "blender": (2, 83, 0),
    "location": "View3D > Edit Mode > Toolbar",
    "description": "Draw Voxel mesh in Edit Mode",
    "warning": "Made purely for fun, don't expect stuff to work ;)",
    "doc_url": "",
    "category": "Add Mesh",
}
    
import bpy
from bpy_extras.view3d_utils import region_2d_to_location_3d, region_2d_to_vector_3d, region_2d_to_origin_3d

import gpu
from gpu_extras.batch import batch_for_shader
import blf
import bgl

import mathutils
from decimal import Decimal, ROUND_HALF_UP
from mathutils import Vector
import bmesh
from math import sin, cos, pi, degrees, floor
import numpy as np
import time
import sys
from bpy.utils.toolsystem import ToolDef
from mathutils import Matrix, Vector

from gpu_extras.presets import draw_circle_2d
from bpy.types import (
    GizmoGroup,
)

axis_items = [
    ("XY", "XY", "", 1),
    ("YZ", "YZ", "", 2),
    ("XZ", "XZ", "", 3),
]        

toolVoxel = "3D View Tool: Edit, Voxel Draw"

shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
xy = None
prev_xy = None
prev_loc = None
area_ptr = None


def get_location(x, y, context):
    region = context.region
    rv3d = context.space_data.region_3d
    loc = (x, y)
    vec = region_2d_to_vector_3d(region, rv3d, loc)   
    
    viewlayer = context.view_layer
    view_vector = region_2d_to_vector_3d(region, rv3d, loc)
    ray_origin = region_2d_to_origin_3d(region, rv3d, loc)
    
    result, location, normal, index, obj, matrix = context.scene.ray_cast(viewlayer, ray_origin, view_vector)
    
    tool = context.workspace.tools.from_space_view3d_mode(bpy.context.mode)
    props = tool.operator_properties('view3d.voxel')
    voxel_size = props.voxel_size
    
    
    if not result:
        if context.scene.voxel_settings.axis == 'XY':
            matrix = Matrix.Rotation(pi/2, 4, 'Z')
        elif context.scene.voxel_settings.axis == 'YZ':
            matrix = Matrix.Rotation(pi/2, 4, 'Y')
        else:
            matrix = Matrix.Rotation(pi/2, 4, 'X')
        
        coords = [Vector((0,0,context.scene.voxel_settings.z)),Vector((0,1,context.scene.voxel_settings.z)),Vector((1,0,context.scene.voxel_settings.z))]
        for i, coord in enumerate(coords):
            coords[i] = matrix @ coord
            
        location = mathutils.geometry.intersect_ray_tri(coords[0], coords[1], coords[2], view_vector,ray_origin,False )

        if location is None:
            return None, -1, None
    else:
        location = [int(Decimal(round(x/voxel_size, 4)).to_integral_value(rounding=ROUND_HALF_UP)) * voxel_size for x in location]

    
    
    location = [int(Decimal(round(x / voxel_size, 4)).to_integral_value(rounding=ROUND_HALF_UP)) * voxel_size for x in location]
    if result:
        if location[0] > 0 and normal.x < -0.9:
            location[0] -= voxel_size
        if location[1] > 0 and normal.y < -0.9:
            location[1] -= voxel_size
        if location[2] > 0 and normal.z < -0.9:
            location[2] -= voxel_size
        
        if location[0] < 0 and normal.x > 0.9:
            location[0] += voxel_size
        if location[1] < 0 and normal.y > 0.9:
            location[1] += voxel_size
        if location[2] < 0 and normal.z > 0.9:
            location[2] += voxel_size
    return location, index, obj



def draw_callback_px():
    context = bpy.context
    
    global area_ptr
    if area_ptr != context.area.as_pointer():
        return
    
    global xy, prev_xy, prev_loc
    if xy is None:
        return
    
    x, y = xy
    tool = context.workspace.tools.from_space_view3d_mode(context.mode)
    if tool.idname not in ['voxel.draw_tool', 'voxel.draw_cube']:
        return
 
    if xy == prev_xy:
        loc = prev_loc
    else:
        loc, _, _ = get_location(x, y, context)
        
    if loc is not None:
        prev_loc = loc
    prev_xy = xy
    
    if loc is None:
        return
    
    
    transform = Matrix.Translation((loc))
    
    bgl.glEnable(bgl.GL_BLEND)
    
    tool = context.workspace.tools.from_space_view3d_mode(bpy.context.mode)
    props = tool.operator_properties('view3d.voxel')
    voxel_size = props.voxel_size / 2
    
    coords = [
        (-voxel_size, -voxel_size, -voxel_size), (+voxel_size, -voxel_size, -voxel_size),
        (-voxel_size, +voxel_size, -voxel_size), (+voxel_size, +voxel_size, -voxel_size),
        (-voxel_size, -voxel_size, +voxel_size), (+voxel_size, -voxel_size, +voxel_size),
        (-voxel_size, +voxel_size, +voxel_size), (+voxel_size, +voxel_size, +voxel_size)]
    
    if context.scene.voxel_settings.axis == 'XY':
        matrix = Matrix.Rotation(pi/2, 4, 'Z')
    elif context.scene.voxel_settings.axis == 'YZ':
        matrix = Matrix.Rotation(pi/2, 4, 'Y')
    else:
        matrix = Matrix.Rotation(pi/2, 4, 'X')

    coords_plane = [
        (-100, +100, context.scene.voxel_settings.z),
        (+100, +100, context.scene.voxel_settings.z),
        (+100, -100, context.scene.voxel_settings.z),
        (-100, -100, context.scene.voxel_settings.z)]
    for i, coord in enumerate(coords_plane):
        coords_plane[i] = matrix @ Vector(coord)
        
    plane_indices = ((0, 1, 2), (2, 3, 0))
    line_indices = (
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7))
    
    
    
    for i, coord in enumerate(coords):
        coords[i] = transform @ Vector(coord)
    batch_lines = batch_for_shader(shader, 'LINES', {'pos': coords}, indices=line_indices)
    batch_plane = batch_for_shader(shader, 'TRIS', {'pos': coords_plane}, indices=plane_indices)
    
    shader.bind()
    shader.uniform_float('color', (1, 0, 0, 1))
    batch_lines.draw(shader)
    
    if props.show_floor:
        shader.uniform_float('color', (1, 1, 0, 0.05))
        batch_plane.draw(shader)

    # restore opengl defaults
    bgl.glLineWidth(1)
    bgl.glDisable(bgl.GL_BLEND)
    
class VoxelProperties(bpy.types.PropertyGroup):
    def update_axis(self, context):
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.editmode_toggle()

    
    z: bpy.props.FloatProperty(name="Z", default=0)
    axis: bpy.props.EnumProperty(name="Floor", default='XY', items=axis_items, update=update_axis)
    


class VoxelPlaneGizmo(GizmoGroup):
    bl_idname = "MESH_GGT_voxel_plane"
    bl_label = "Voxel Grid Plane"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        return context.workspace.tools.from_space_view3d_mode(bpy.context.mode).idname in ['voxel.draw_tool', 'voxel.draw_cube']

    def setup(self, context):
        mpr = self.gizmos.new("GIZMO_GT_arrow_3d")
        mpr.target_set_prop("offset", context.scene.voxel_settings, "z")
        
        mpr.line_width = 1

        mpr.color = 1, 1, 0
        mpr.alpha = 0.5

        mpr.color_highlight = 1.0, 1, 0
        mpr.alpha_highlight = 1.0

        self.roll_widget = mpr

    def refresh(self, context):
        mpr = self.roll_widget
        
        if context.scene.voxel_settings.axis == 'XY':
            matrix = Matrix.Rotation(pi/2, 4, 'Z')
        elif context.scene.voxel_settings.axis == 'YZ':
            matrix = Matrix.Rotation(pi/2, 4, 'Y')
        else:
            matrix = Matrix.Rotation(pi/2, 4, 'X')

        mpr.matrix_basis = matrix.normalized()
        


    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':

            context.window_manager.modal_handler_add(self)
            self.obj = context.object
            self.me = self.obj.data
            self.bm = bmesh.from_edit_mesh(self.me)
            
            loc, index, obj = get_location(event.mouse_region_x, event.mouse_region_y, context)
            if loc is None:
                return {'CANCELLED'}
            global start_pos, cube_drawing
            cube_drawing = True
            start_pos = Vector(loc)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}



class VoxalDrawOperator(bpy.types.Operator):
    """Draw 3D mesh"""
    bl_idname = "view3d.voxel"
    bl_label = "Voxel Draw"
    bl_options = {"REGISTER", "UNDO"}
    
    
    voxel_size: bpy.props.FloatProperty(name='Voxel Size', default=1, min=0.001)
    show_floor: bpy.props.BoolProperty(name = 'Show Floor', default=False)
    
    def add_voxel(self, context, event, delete=False):
        loc, index, obj = get_location(event.mouse_region_x, event.mouse_region_y, context)
        
        if loc is None:
            return
        
        loc = context.object.matrix_world.inverted() @ Vector(loc)
        if delete:
            if obj != context.object:
                return
            bmesh.ops.delete(self.bm, geom=[self.bm.faces[index]], context='FACES')
        else:
            transform = Matrix.Translation((loc))
            verts = bmesh.ops.create_cube(self.bm, size=self.voxel_size, matrix=transform)
        bmesh.update_edit_mesh(self.me)
    
    def modal(self, context, event):
        context.area.tag_redraw()
        
        if event.type == 'MOUSEMOVE':
            self.add_voxel(context, event, event.ctrl)
            return {'RUNNING_MODAL'}
    
        elif event.type == 'LEFTMOUSE':
            bmesh.ops.remove_doubles(self.bm, verts=self.bm.verts,dist=0.001)
            bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces)
            bmesh.update_edit_mesh(self.me)
            return {'FINISHED'}
        
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


    def invoke(self, context, event):
        if context.area.type == 'VIEW_3D':

            context.window_manager.modal_handler_add(self)
            self.obj = context.object
            self.me = self.obj.data
            self.bm = bmesh.from_edit_mesh(self.me)
            
            self.add_voxel(context, event, event.ctrl)
            return {'RUNNING_MODAL'}
        else:
            self.report({'WARNING'}, "View3D not found, cannot run operator")
            return {'CANCELLED'}


@ToolDef.from_fn
def toolVoxelDraw():
    
    def draw_settings(context, layout, tool):
        props = tool.operator_properties('view3d.voxel')
        layout.prop(props, 'voxel_size')
        layout.prop(context.scene.voxel_settings, 'axis')
        layout.prop(props, 'show_floor')
            
    def draw_cursor(context, tool, loc):
        global xy, prev_xy, area_ptr
        xy = loc
        xy = (xy[0] - context.area.x, xy[1] - context.area.y)
        
        area_ptr = context.area.as_pointer()
        if xy[0] < 0 or xy[1] < 0:
            return
        
        context.area.tag_redraw()
    
    return dict(idname = "voxel.draw_tool",
        label = "Draw Voxel",
        description = "Draw 3D Voxel mesh in Edit Mode",
        icon = "ops.mesh.primitive_cube_add_gizmo",
        widget = None,
        keymap = toolVoxel,
        draw_settings = draw_settings,
        draw_cursor = draw_cursor
        )


def getToolList(spaceType, contextMode):
    from bl_ui.space_toolsystem_common import ToolSelectPanelHelper
    cls = ToolSelectPanelHelper._tool_class_from_space_type(spaceType)
    return cls._tools[contextMode]

def registerVoxel(cls):
    tools = getToolList('VIEW_3D', 'EDIT_MESH')
    tools += None, cls
    del tools


def unregisterVoxel(cls):
    tools = getToolList('VIEW_3D', 'EDIT_MESH')

    index = tools.index(cls) - 1 #None
    tools.pop(index)
    tools.remove(cls)
    del tools


keymapDraw = (toolVoxel,
        {"space_type": 'VIEW_3D', "region_type": 'WINDOW'},
        {"items": [
            ("view3d.voxel", {"type": 'LEFTMOUSE', "value": 'PRESS'},
             {"properties": []}),
             ("view3d.voxel", {"type": 'LEFTMOUSE', "value": 'PRESS', "ctrl": True},
             {"properties": []}),
        ]},)

emptyKeymapDraw = (toolVoxel,
        {"space_type": 'VIEW_3D', "region_type": 'WINDOW'},
        {"items": []},)




def registerVoxelKeymaps(keymapDraw, emptyKeymapDraw):
    keyconfigs = bpy.context.window_manager.keyconfigs
    kc_defaultconf = keyconfigs.default
    kc_addonconf = keyconfigs.addon

    from bl_keymap_utils.io import keyconfig_init_from_data
    keyconfig_init_from_data(kc_defaultconf, [emptyKeymapDraw])

    keyconfig_init_from_data(kc_addonconf, [keymapDraw])

def unregisterVoxelKeymaps(keymapDraw):
    keyconfigs = bpy.context.window_manager.keyconfigs
    defaultmap = keyconfigs.get("blender").keymaps
    addonmap   = keyconfigs.get("blender addon").keymaps

    for km_name, km_args, km_content in [keymapDraw]:
        keymap = addonmap.find(km_name, **km_args)
        keymap_items = keymap.keymap_items
        for item in km_content['items']:
            item_id = keymap_items.find(item[0])
            if item_id != -1:
                keymap_items.remove(keymap_items[item_id])
        addonmap.remove(keymap)
        defaultmap.remove(defaultmap.find(km_name, **km_args))

classes = [
    VoxalDrawOperator,
    VoxelProperties,
    VoxelPlaneGizmo
]

_handle = None

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
    global _handle
    _handle = bpy.types.SpaceView3D.draw_handler_add(draw_callback_px, (), 'WINDOW', 'POST_VIEW')
    
    bpy.types.Scene.voxel_settings = bpy.props.PointerProperty(type=VoxelProperties)
    registerVoxel(toolVoxelDraw)
    registerVoxelKeymaps(keymapDraw, emptyKeymapDraw)

def unregister():
    unregisterVoxelKeymaps(keymapDraw)
    unregisterVoxel(toolVoxelDraw)
    
    bpy.types.SpaceView3D.draw_handler_remove(_handle, 'WINDOW')

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()