import bpy
import sys
import numpy as np
import torch
from os import path as osp
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

bl_info = {
    "name": "VPoser for Blender",
    "blender": (2, 80, 0),
    "category": "SMPL-X",
}

# Load VPoser
DIR_PATH = osp.dirname(osp.realpath(__file__))
VP, _ = load_model(DIR_PATH + '/data/V02_05', model_code=VPoser,
                   remove_words_in_model_weights='vp_model.',
                   disable_grad=True)

class ObjectVPoserUpdatePose(bpy.types.Operator):
    """Update pose by VPoser"""
    bl_idname = "object.vposer_update_pose"
    bl_label = "Update pose by VPoser"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if 'smplx_blender_addon' not in sys.modules:
            self.report({"ERROR"}, "No smplx_blender_addon")
            return {"CANCELLED"}

        smplx = sys.modules['smplx_blender_addon']
        body_bones_names = \
            smplx.SMPLX_JOINT_NAMES[1 : smplx.NUM_SMPLX_BODYJOINTS + 1]
        armature = context.object

        if armature.type != 'ARMATURE':
            self.report({"ERROR"}, "Current object is not armature")
            return {"CANCELLED"}

        body_pose = []

        for bone_name in body_bones_names:
            body_pose.append(smplx.rodrigues_from_pose(armature, bone_name))

        body_pose = torch.Tensor(body_pose).reshape((1,-1)).type(torch.float)
        body_poZ = VP.encode(body_pose).mean
        body_pose_rec = \
            VP.decode(body_poZ)['pose_body'].contiguous().view(-1, 63)

        for i, bone_name in enumerate(body_bones_names):
            smplx.set_pose_from_rodrigues(armature, bone_name, 
                                          body_pose_rec[0][3*i : 3*(i+1)])

        bpy.ops.object.smplx_set_poseshapes()

        return {'FINISHED'}

class ObjectVPoserGeneratePose(bpy.types.Operator):
    """Generate random pose by VPoser"""
    bl_idname = "object.vposer_generate_pose"
    bl_label = "Generate random pose by VPoser"
    bl_options = {'REGISTER', 'UNDO'}


    def execute(self, context):
        if 'smplx_blender_addon' not in sys.modules:
            self.report({"ERROR"}, "No smplx_blender_addon")
            return {"CANCELLED"}

        smplx = sys.modules['smplx_blender_addon']
        body_bones_names = \
            smplx.SMPLX_JOINT_NAMES[1 : smplx.NUM_SMPLX_BODYJOINTS + 1]
        armature = context.object

        if armature.type != 'ARMATURE':
            self.report({"ERROR"}, "Current object is not armature")
            return {"CANCELLED"}


        body_poZ_sample = \
            torch.from_numpy(np.random.randn(1,32).astype(np.float32))
        body_pose = \
            VP.decode(body_poZ_sample)['pose_body'].contiguous().view(-1, 63)

        for i, bone_name in enumerate(body_bones_names):
            smplx.set_pose_from_rodrigues(armature, bone_name, 
                                          body_pose[0][3*i : 3*(i+1)])

        bpy.ops.object.smplx_set_poseshapes()

        return {'FINISHED'}

class SMPLX_PT_VPoser(bpy.types.Panel):
    bl_label = "VPoser"
    bl_category = "SMPL-X"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        
        col.operator("object.vposer_update_pose", text="Update")
        col.operator("object.vposer_generate_pose", text="Generate")

def register():
    bpy.utils.register_class(ObjectVPoserUpdatePose)
    bpy.utils.register_class(ObjectVPoserGeneratePose)
    bpy.utils.register_class(SMPLX_PT_VPoser)

def unregister():
    bpy.utils.unregister_class(ObjectVPoserPoseUpdate)
    bpy.utils.unregister_class(ObjectVPoserGeneratePose)
    bpy.utils.unregister_class(SMPLX_PT_VPoser)

if __name__ == "__main__":
    register()
