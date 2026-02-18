# SPDX-FileCopyrightText: 2025 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

bl_info = {
    "name": "GDSII Importer with PDK Support",
    "author": "aesc silicon",
    "version": (1, 0, 0),
    "blender": (3, 4, 0),
    "location": "File > Import",
    "description": "Import GDSII files with PDK layer stack support",
    "warning": "Requires gdstk and PyYAML packages",
    "doc_url": "",
    "category": "Import-Export",
}

import bpy
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper
import os
from pathlib import Path

# Try to import required packages
try:
    import gdstk
    import numpy as np
    import yaml
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    IMPORT_ERROR = str(e)


# ============================================================================
# PDK DEFINITIONS
# ============================================================================

# PDK Configuration paths
PDK_CONFIGS = {
    'IHP_SG13G2': {
        'name': 'IHP Open PDK (SG13G2)',
        'config_path': 'configs/ihp-sg13g2.yaml',
        'color_path': 'configs/colors/ihp-sg13g2/'
    },
    'IHP_SG13CMOS5L': {
        'name': 'IHP Open PDK (SG13CMOS5L)',
        'config_path': 'configs/ihp-sg13cmos5l.yaml',
        'color_path': 'configs/colors/ihp-sg13cmos5l/'
    },
    'SKY130': {
        'name': 'SkyWater SKY130 PDK',
        'config_path': 'configs/sky130.yaml',
        'color_path': 'configs/colors/sky130/'
    },
    'GF180MCU': {
        'name': 'GlobalFoundries GF180MCU PDK',
        'config_path': 'configs/gf180mcu.yaml',
        'color_path': 'configs/colors/gf180mcu/'
    },
    'ASAP7': {
        'name': 'ASAP7 7nm FinFET PDK',
        'config_path': 'configs/asap7.yaml',
        'color_path': 'configs/colors/asap7/'
    },
}


# ============================================================================
# SCENE SETUP FUNCTIONS
# ============================================================================

def setup_chip_scene(chip_x, chip_y, collection=None):
    """Initialize scene with camera, light, and chip base"""
    # Determine which collection to use
    target_collection = collection if collection is not None else bpy.context.collection

    # --------------------------
    # SETUP WORLD
    # --------------------------
    world = bpy.data.worlds.new("ChipWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.05, 0.05, 0.08, 1.0)  # Dark bluish background
    bg.inputs[1].default_value = 1.0

    # --------------------------
    # SETUP LIGHTING
    # --------------------------
    # Add a Sun light
    sun = bpy.data.lights.new(name="Sun", type='SUN')
    sun_obj = bpy.data.objects.new("Sun", sun)
    target_collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (0.8, 0.0, 0.8)  # Slight angle
    sun.energy = 3.0

    # Add soft shadows
    if bpy.app.version < (4, 0, 0):
        # Blender 3.x - use global setting
        bpy.context.scene.eevee.use_soft_shadows = True
    else:
        # Blender 4.0+ - soft shadows are default in EEVEE Next
        # Set shadow properties on the light instead
        sun.use_shadow = True

    # --------------------------
    # SETUP CAMERA
    # --------------------------
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    target_collection.objects.link(cam)
    bpy.context.scene.camera = cam

    # Position camera above the chip
    cam.location = (chip_x / 2, chip_y / 2, 200)
    cam.rotation_euler = (0, 0, 0)

    # --------------------------
    # CREATE CHIP BASE
    # --------------------------
    mesh = bpy.data.meshes.new("ChipBaseMesh")
    chip_base = bpy.data.objects.new("ChipBase", mesh)
    target_collection.objects.link(chip_base)

    # Define vertices for a plane of correct size
    verts = [(0, 0, -0.005),
             (0, chip_y, -0.005),
             (chip_x, chip_y, -0.005),
             (chip_x, 0, -0.005)]
    faces = [(0, 1, 2, 3)]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Assign material
    mat = bpy.data.materials.new(name="ChipBaseMat")
    if bpy.app.version >= (4, 0, 0):
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.05, 0.07, 0.1, 1)
    else:
        mat.diffuse_color = (0.05, 0.07, 0.1, 1)
    chip_base.data.materials.append(mat)

    print("✓ Chip scene setup complete!")

# ============================================================================
# MATERIAL AND MESH CREATION FUNCTIONS
# ============================================================================

def create_material(name, color):
    """Create a material with given color"""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create nodes
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_output = nodes.new(type='ShaderNodeOutputMaterial')

    # Set color and properties
    node_bsdf.inputs['Base Color'].default_value = color.get('color', [1.0, 1.0, 1.0, 1.0])
    node_bsdf.inputs['Metallic'].default_value = color.get('metallic', 0.1)
    node_bsdf.inputs['Roughness'].default_value = color.get('roughness', 0.5)

    if 'Alpha' in node_bsdf.inputs:
        node_bsdf.inputs['Alpha'].default_value = color.get('color', [1.0, 1.0, 1.0, 1.0])[-1]

    # Link nodes
    mat.node_tree.links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])

    return mat


def _extract_polygons(gds_path, layer, unit, crop_box):
    """Read, flatten, and optionally crop GDS polygons for a single layer.

    Returns a plain list of gdstk.Polygon objects suitable for use as
    boolean operands (e.g. as wrap_polygons in create_extruded_layer).
    """
    library = gdstk.read_gds(gds_path, unit=unit, filter={layer})
    merged = gdstk.Cell("_EXTRACT")
    for cell in library.top_level():
        flat = cell.flatten()
        merged.add(*flat.polygons)
        for path in flat.paths:
            merged.add(*path.to_polygons())
    if crop_box is not None:
        x_min, y_min, x_max, y_max = crop_box
        crop_rect = gdstk.rectangle((x_min, y_min), (x_max, y_max))
        cropped = []
        for poly in merged.polygons:
            cropped.extend(gdstk.boolean(poly, crop_rect, 'and'))
        return cropped
    return list(merged.polygons)


def _add_extruded_polygon(polygon, z_bot, z_top, all_verts, all_faces, v_offset):
    """Append one extruded polygon's verts and faces; returns updated v_offset."""
    points = polygon.points
    n = len(points)
    if n < 3:
        return v_offset
    bottom = np.column_stack([points, np.full(n, z_bot)])
    top    = np.column_stack([points, np.full(n, z_top)])
    all_verts.extend(np.vstack([bottom, top]).tolist())
    all_faces.append([v_offset + j for j in range(n)])
    all_faces.append([v_offset + 2*n-1-j for j in range(n)])
    all_faces.extend([[v_offset + j, v_offset + (j+1)%n,
                       v_offset + (j+1)%n + n, v_offset + j + n]
                      for j in range(n)])
    return v_offset + 2 * n


def create_extruded_layer(report, gds_path, z, height, layer, name, color, unit=1e-6, crop_box=None,
                          wrap_polygons=None, wrap_z_bottom=None):
    """Create extruded geometry for a specific GDS layer.

    When wrap_polygons and wrap_z_bottom are supplied (wrap_around feature),
    the layer polygons are split into two groups via boolean intersection:
      - regions that do NOT overlap wrap_polygons → extruded at the normal z/height
      - regions that DO overlap wrap_polygons     → extruded from wrap_z_bottom to
                                                    z+height, so the layer visually
                                                    penetrates through the reference
                                                    geometry (e.g. gate through fins)
    """
    # Read and filter GDS
    library = gdstk.read_gds(gds_path, unit=unit, filter={layer})

    # Merge and flatten all top-level cells
    merged_cell = gdstk.Cell("MERGED")
    for cell in library.top_level():
        flattened = cell.flatten()
        merged_cell.add(*flattened.polygons)
        for path in flattened.paths:
            merged_cell.add(*path.to_polygons())

    # Apply crop if specified
    if crop_box is not None:
        x_min, y_min, x_max, y_max = crop_box
        crop_rect = gdstk.rectangle((x_min, y_min), (x_max, y_max))
        cropped_polygons = []
        for polygon in merged_cell.polygons:
            cropped_polygons.extend(gdstk.boolean(polygon, crop_rect, "and"))
        merged_cell = gdstk.Cell("MERGED")
        merged_cell.add(*cropped_polygons)

    polygon_count = len(merged_cell.polygons)
    if polygon_count == 0:
        print(f"⚠ Layer {name}: No geometry found")
        return None

    all_verts = []
    all_faces = []
    v_offset = 0

    if wrap_polygons:
        # Split: regions that cross the reference layer get extended z; others stay normal
        polys_normal = gdstk.boolean(merged_cell.polygons, wrap_polygons, 'not')
        polys_wrap   = gdstk.boolean(merged_cell.polygons, wrap_polygons, 'and')
        for poly in polys_normal:
            v_offset = _add_extruded_polygon(poly, z, z + height, all_verts, all_faces, v_offset)
        for poly in polys_wrap:
            v_offset = _add_extruded_polygon(poly, wrap_z_bottom, z + height, all_verts, all_faces, v_offset)
    else:
        for polygon in merged_cell.polygons:
            v_offset = _add_extruded_polygon(polygon, z, z + height, all_verts, all_faces, v_offset)

    # Create mesh
    mesh = bpy.data.meshes.new(name=f"M{name}")
    mesh.from_pydata(all_verts, [], all_faces)
    mesh.update()

    obj = bpy.data.objects.new(name=f"L{name}", object_data=mesh)
    bpy.context.collection.objects.link(obj)

    # Apply material
    mat = create_material(f"Mat_{name}", color)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    print(f"✓ {name}: {polygon_count} polygons, {len(all_verts)} vertices")
    report({'INFO'}, f"✓ {name}: {polygon_count} polygons, {len(all_verts)} vertices")
    return obj


# ============================================================================
# PRE-IMPORT PDK SELECTION DIALOG
# ============================================================================

def get_color_schemes(self, context):
    """Dynamically generate color scheme list based on selected PDK"""
    items = []

    pdk = getattr(context.scene, 'gdsii_pdk_selection', 'IHP_SG13G2')
    addon_dir = Path(__file__).parent
    color_path = addon_dir / PDK_CONFIGS.get(pdk, {}).get('color_path', pdk)
    schemes = [('realistic', 'Realistic', 'Realistic color scheme')]
    for file in color_path.glob('*yaml'):
        # Make sure realistic is the default choice
        if file.stem == 'realistic':
            continue
        color_file = yaml.safe_load(file.read_text(encoding='utf-8'))
        schemes.append((file.stem, color_file.get('name', file.stem),
                       color_file.get('description', file.stem)))
    return schemes


class GDSIIPreImportDialog(bpy.types.Operator):
    """Select PDK before importing GDSII file"""
    bl_idname = "import_scene.gdsii_pdk_dialog"
    bl_label = "Select PDK for GDSII Import"
    bl_options = {'REGISTER'}

    pdk_selection: EnumProperty(
        name="PDK",
        description="Process Design Kit to use for layer stack",
        items=[
            ('IHP_SG13G2', "IHP Open PDK SG13G2", "IHP SG13G2 130nm BiCMOS process"),
            ('IHP_SG13CMOS5L', "IHP Open PDK SG13CMOS5L", "IHP SG13CMOS5L 130nm CMOS5L process"),
            ('SKY130', "SkyWater SKY130 PDK", "SkyWater SKY130 130nm process"),
            ('GF180MCU', "GlobalFoundries GF180MCU PDK", "GlobalFoundries GF180MCU 180nm process"),
            ('ASAP7', "ASAP7 7nm FinFET PDK", "ASAP7 7nm FinFET research PDK"),
        ],
        default='IHP_SG13G2',
    )

    config_path: StringProperty(
        name="Config Path",
        description="Path to PDK YAML configuration file",
        default="",
        subtype='FILE_PATH',
    )

    use_custom_config: BoolProperty(
        name="Use Custom Config",
        description="Use a custom YAML config file instead of built-in PDK",
        default=False,
    )

    def invoke(self, context, event):
        # Set default config path based on PDK selection
        self.update_config_path()
        return context.window_manager.invoke_props_dialog(self, width=400)

    def update_config_path(self):
        """Update config path based on selected PDK"""
        if not self.use_custom_config and self.pdk_selection in PDK_CONFIGS:
            pdk_info = PDK_CONFIGS[self.pdk_selection]
            # Try to find config relative to add-on directory
            addon_dir = Path(__file__).parent
            config_file = addon_dir / pdk_info['config_path']
            if config_file.exists():
                self.config_path = str(config_file)

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Process Design Kit Selection", icon='PRESET')
        box.prop(self, "pdk_selection")

        box = layout.box()
        box.label(text="Configuration File", icon='FILE')
        box.prop(self, "use_custom_config")
        if self.use_custom_config:
            box.prop(self, "config_path")
        else:
            pdk_info = PDK_CONFIGS.get(self.pdk_selection, {})
            box.label(text=f"Using: {pdk_info.get('config_path', 'N/A')}", icon='INFO')

    def execute(self, context):
        # Store PDK settings in scene for the importer to use
        context.scene.gdsii_pdk_selection = self.pdk_selection
        context.scene.gdsii_config_path = self.config_path
        context.scene.gdsii_use_custom_config = self.use_custom_config

        # Open the file browser for GDSII import
        bpy.ops.import_scene.gdsii('INVOKE_DEFAULT')
        return {'FINISHED'}


# ============================================================================
# GDSII IMPORTER
# ============================================================================

class ImportGDSII(bpy.types.Operator, ImportHelper):
    """Import GDSII Layout File"""
    bl_idname = "import_scene.gdsii"
    bl_label = "Import GDSII"
    bl_options = {'PRESET', 'UNDO'}

    # File browser filter
    filename_ext = ".gds"
    filter_glob: StringProperty(
        default="*.gds;*.gdsii",
        options={'HIDDEN'},
    )

    # Import options
    unit_scale: FloatProperty(
        name="Unit Scale",
        description="GDS database unit scale (typically 1e-6 for micrometers)",
        default=1e-6,
        min=1e-12,
        max=1.0,
    )

    z_scale: FloatProperty(
        name="Z Scale",
        description="Scale factor for layer heights",
        default=1.0,
        min=0.001,
        max=1000.0,
    )

    create_collection: BoolProperty(
        name="Create Collection",
        description="Import layers into a new collection",
        default=True,
    )

    # Scene setup
    setup_scene: BoolProperty(
        name="Setup Scene",
        description="Initialize scene with camera, light, and chip base",
        default=True,
    )

    # Color schemes
    color_scheme: EnumProperty(
        name="Color Scheme",
        description="Select color scheme for the PDK",
        items=get_color_schemes
    )

    # Crop region options
    use_crop: BoolProperty(
        name="Crop to Region",
        description="Import only a specific region of the chip",
        default=False,
    )

    # Add metal dummy fill
    add_fill: BoolProperty(
        name="Metal dumm fill",
        description="Include metal dummy fill",
        default=False,
    )

    crop_x: FloatProperty(
        name="X",
        description="X coordinate of crop region (lower-left corner)",
        default=0.0,
    )

    crop_y: FloatProperty(
        name="Y",
        description="Y coordinate of crop region (lower-left corner)",
        default=0.0,
    )

    crop_width: FloatProperty(
        name="Width",
        description="Width of crop region",
        default=1000.0,
        min=0.1,
    )

    crop_height: FloatProperty(
        name="Height",
        description="Height of crop region",
        default=1000.0,
        min=0.1,
    )

    def draw(self, context):
        """Custom layout for the file browser sidebar"""
        layout = self.layout

        box = layout.box()
        box.label(text="Import Settings:", icon='IMPORT')
        box.prop(self, "unit_scale")
        box.prop(self, "z_scale")
        box.prop(self, "create_collection")

        # Scene setup
        box = layout.box()
        box.label(text="Scene Setup:", icon='SCENE_DATA')
        box.prop(self, "setup_scene")

        # Color scheme
        box = layout.box()
        box.label(text="Color Scheme:", icon='SCENE_DATA')
        box.prop(self, "color_scheme")

        # Add metal dummy fill
        box = layout.box()
        box.label(text="Metal Dummy Fill:", icon='SCENE_DATA')
        box.prop(self, "add_fill")

        # Crop region
        box = layout.box()
        box.label(text="Crop Region:", icon='BORDERMOVE')
        box.prop(self, "use_crop")
        if self.use_crop:
            row = box.row(align=True)
            row.prop(self, "crop_x")
            row.prop(self, "crop_y")
            row = box.row(align=True)
            row.prop(self, "crop_width")
            row.prop(self, "crop_height")

        # Show selected PDK
        box = layout.box()
        box.label(text="Selected PDK:", icon='PRESET')
        pdk = getattr(context.scene, 'gdsii_pdk_selection', 'IHP_SG13G2')
        pdk_name = PDK_CONFIGS.get(pdk, {}).get('name', pdk)
        box.label(text=pdk_name)

    def execute(self, context):
        if not DEPENDENCIES_OK:
            self.report({'ERROR'}, f"Missing dependencies: {IMPORT_ERROR}. Install gdstk, numpy, and PyYAML.")
            return {'CANCELLED'}

        return self.import_gdsii(context, self.filepath)

    def import_gdsii(self, context, filepath):
        """Main import function"""
        try:
            # Get PDK settings from scene
            pdk_selection = getattr(context.scene, 'gdsii_pdk_selection', 'IHP_SG13G2')
            config_path = getattr(context.scene, 'gdsii_config_path', '')
            use_custom = getattr(context.scene, 'gdsii_use_custom_config', False)

            # Determine config file path
            pdk_info = PDK_CONFIGS.get(pdk_selection, {})
            addon_dir = Path(__file__).parent
            if use_custom and config_path:
                yamlfile = Path(config_path)
            else:
                # Use built-in PDK config
                yamlfile = addon_dir / pdk_info.get('config_path', 'configs/ihp-sg13g2.yaml')

            # Load layer stack configuration
            if not yamlfile.exists():
                self.report({'ERROR'}, f"Layer stack file not found: {yamlfile}")
                return {'CANCELLED'}

            layerstack = yaml.safe_load(yamlfile.read_text(encoding='utf-8'))

            # Setup crop box if enabled
            crop_box = None
            if self.use_crop:
                # Convert to GDS units (typically micrometers)
                crop_box = (
                    self.crop_x,
                    self.crop_y,
                    self.crop_x + self.crop_width,
                    self.crop_y + self.crop_height
                )
                chip_width = self.crop_width
                chip_height = self.crop_height
                print(f"Cropping to region: X={self.crop_x}, Y={self.crop_y}, W={self.crop_width}, H={self.crop_height}")
            else:
                # Determine chip dimensions for scene setup
                lib = gdstk.read_gds(filepath, unit=self.unit_scale)
                bboxes = [c.bounding_box() for c in lib.top_level() if c.bounding_box() is not None]
                xmin = min(b[0][0] for b in bboxes)
                ymin = min(b[0][1] for b in bboxes)
                xmax = max(b[1][0] for b in bboxes)
                ymax = max(b[1][1] for b in bboxes)
                chip_width = xmax - xmin
                chip_height = ymax - ymin

            # Create collection for imported layers
            collection = None
            if self.create_collection:
                col_name = Path(filepath).stem
                collection = bpy.data.collections.new(col_name)
                context.scene.collection.children.link(collection)
                # Make it active
                layer_collection = context.view_layer.layer_collection.children[collection.name]
                context.view_layer.active_layer_collection = layer_collection

            # Setup scene if requested
            if self.setup_scene:
                setup_chip_scene(chip_width, chip_height, collection)

            colorfile = addon_dir / pdk_info.get('color_path', 'configs/colors/ihp-sg13g2') / f"{self.color_scheme}.yaml"
            color_file = yaml.safe_load(colorfile.read_text(encoding='utf-8'))

            print(f"Starting GDS import from: {filepath}")
            print(f"Using PDK: {pdk_selection}")
            print(f"Layer stack config: {yamlfile}")

            # Import each layer from the stack
            imported_count = 0
            for layer_name, data in layerstack.items():
                z = data['z'] * self.z_scale
                height = data['height'] * self.z_scale
                layer_index = (data['index'], data['type'])
                if data.get('purpose', 'drawing') == 'filler' and not self.add_fill:
                    continue

                layer_cfg = color_file.get('layers', {}).get(layer_name, {})

                # Resolve wrap_around: extract reference layer polygons so the
                # gate (or any other layer) can be split at fin/reference boundaries
                wrap_polygons = None
                wrap_z_bottom = None
                if 'wrap_around' in data:
                    wa = data['wrap_around']
                    target_name = wa['layer']
                    z_extend = wa.get('z_extend', 0.0) * self.z_scale
                    target_data = layerstack.get(target_name)
                    if target_data:
                        target_layer = (target_data['index'], target_data['type'])
                        wrap_polygons = _extract_polygons(filepath, target_layer, self.unit_scale, crop_box)
                        wrap_z_bottom = target_data['z'] * self.z_scale - z_extend
                    else:
                        print(f"⚠ wrap_around: layer '{target_name}' not found in stack, skipping wrap for {layer_name}")

                obj = create_extruded_layer(
                    self.report,
                    filepath,
                    z,
                    height,
                    layer_index,
                    layer_name,
                    layer_cfg,
                    unit=self.unit_scale,
                    crop_box=crop_box,
                    wrap_polygons=wrap_polygons,
                    wrap_z_bottom=wrap_z_bottom,
                )

                if obj is not None:
                    imported_count += 1

            self.report({'INFO'}, f"Imported {imported_count} layers from {Path(filepath).name}")
            print(f"✓ Import complete! {imported_count} layers imported.")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


# ============================================================================
# MENU INTEGRATION
# ============================================================================

def menu_func_import(self, context):
    self.layout.operator(GDSIIPreImportDialog.bl_idname, 
                        text="GDSII (.gds)")


# ============================================================================
# REGISTRATION
# ============================================================================

# Properties to store PDK settings in scene
def register_properties():
    bpy.types.Scene.gdsii_pdk_selection = StringProperty(default='IHP_SG13G2')
    bpy.types.Scene.gdsii_config_path = StringProperty(default='')
    bpy.types.Scene.gdsii_use_custom_config = BoolProperty(default=False)

def unregister_properties():
    del bpy.types.Scene.gdsii_pdk_selection
    del bpy.types.Scene.gdsii_config_path
    del bpy.types.Scene.gdsii_use_custom_config

classes = (
    GDSIIPreImportDialog,
    ImportGDSII,
)

def register():
    # Check dependencies on registration
    if not DEPENDENCIES_OK:
        print(f"⚠ GDSII Importer: Missing dependencies - {IMPORT_ERROR}")
        print("  Install with: pip install gdstk numpy pyyaml --break-system-packages")

    register_properties()

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    unregister_properties()

if __name__ == "__main__":
    register()
