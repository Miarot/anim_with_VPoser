import bpy
import sys
import numpy as np
import torch
from torch import nn
from os import path as osp
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.body_model.body_model import BodyModel

bl_info = {
    "name": "VPoser for Blender",
    "blender": (2, 80, 0),
    "category": "SMPL-X",
}

#Load VPoser
DIR_PATH = osp.dirname(osp.realpath(__file__)) # comment while test
#DIR_PATH = "/home/miaro/itmo/anim/vposer_blender_addon" # for test
BODY_MODELS_PATH =  osp.join(DIR_PATH,"data/models/smplx")
NEUTRAL_MODEL = BodyModel(osp.join(BODY_MODELS_PATH, "SMPLX_NEUTRAL.npz"))
MALE_MODEL = BodyModel(osp.join(BODY_MODELS_PATH, "SMPLX_MALE.npz"))
FEMALE_MODEL = BodyModel(osp.join(BODY_MODELS_PATH, "SMPLX_FEMALE.npz"))
VP, _ = load_model(osp.join(DIR_PATH, "data/V02_05"), 
                   model_code=VPoser,
                   remove_words_in_model_weights='vp_model.',
                   disable_grad=True)

sys.path.insert(0, DIR_PATH)
import src.ik_engine
#from importlib import reload # for test
#reload(src.ik_engine) # for test

def check_compatibility(self, armature):
    if 'smplx_blender_addon' not in sys.modules:
        self.report({"ERROR"}, "No smplx_blender_addon")
        return False

    if armature.type != 'ARMATURE':
        self.report({"ERROR"}, "Current object is not armature")
        return False

    return True

def get_body_parameters(smplx, armature):
    body_bones_names = \
        smplx.SMPLX_JOINT_NAMES[1 : smplx.NUM_SMPLX_BODYJOINTS + 1]

    body_pose = []

    for bone_name in body_bones_names:
        body_pose.append(smplx.rodrigues_from_pose(armature, bone_name))

    body_pose = torch.Tensor(body_pose).reshape((1,-1)).type(torch.float)

    return (body_bones_names, body_pose)


class ObjectVPoserUpdatePose(bpy.types.Operator):
    """Update pose by VPoser"""
    bl_idname = "object.vposer_update_pose"
    bl_label = "Update pose by VPoser"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            if context.active_object.mode == 'POSE':
                return True
            else: 
                return False
        except: return False

    def execute(self, context):
        armature = context.object

        if not check_compatibility(self, armature):
            return {"CANCELLED"}

        smplx = sys.modules['smplx_blender_addon']
        body_bones_names, body_pose = get_body_parameters(smplx, armature)

        body_poZ = VP.encode(body_pose).mean
        body_pose_rec = \
            VP.decode(body_poZ)['pose_body'].contiguous().view(-1, 63)

        for i, bone_name in enumerate(body_bones_names):
            smplx.set_pose_from_rodrigues(armature, bone_name, 
                                          body_pose_rec[0][3*i : 3*(i+1)])

        return {'FINISHED'}

class ObjectVPoserGeneratePose(bpy.types.Operator):
    """Generate random pose by VPoser"""
    bl_idname = "object.vposer_generate_pose"
    bl_label = "Generate random pose by VPoser"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            if context.active_object.mode == 'POSE':
                return True
            else: 
                return False
        except: return False


    def execute(self, context):
        armature = context.object

        if not check_compatibility(self, armature):
            return {"CANCELLED"}

        smplx = sys.modules['smplx_blender_addon']
        body_bones_names, _ = get_body_parameters(smplx, armature)

        body_poZ_sample = \
            torch.from_numpy(np.random.randn(1,32).astype(np.float32))
        body_pose = \
            VP.decode(body_poZ_sample)['pose_body'].contiguous().view(-1, 63)

        for i, bone_name in enumerate(body_bones_names):
            smplx.set_pose_from_rodrigues(armature, bone_name, 
                                          body_pose[0][3*i : 3*(i+1)])

        return {'FINISHED'}

class ObjectVPoserAddIKBone(bpy.types.Operator):
    """Create ik bone for current bone"""
    bl_idname = "object.vposer_add_ik_bone"
    bl_label = "Create ik bone for current bone"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            if context.active_object.mode == 'POSE':
                return True
            else: 
                return False
        except: return False

        
    def execute(self, context):
        armature = context.object

        if not check_compatibility(self, armature):
            return {"CANCELLED"}

        # check if reference object for bones already exist
        is_ref_exist = False
        for child in armature.children:
            if "ik_bone_ref" == child.name:
                is_ref_exist = True

        # crate reference object if it not exist
        if not is_ref_exist:
            ik_bone_ref = bpy.ops.mesh.primitive_ico_sphere_add(
                    radius=0.05,
                    enter_editmode=False, 
                    align='WORLD', 
                    location=(0, 0, 0),
                    scale=(1, 1, 1)
            )

            ik_bone_ref = context.object
            ik_bone_ref.name = "ik_bone_ref"
            ik_bone_ref.parent = armature
            ik_bone_ref.hide_viewport = True
            ik_bone_ref.hide_select = True
            ik_bone_ref.hide_render = True

            # restore context
            ik_bone_ref.select_set(False)
            bpy.context.view_layer.objects.active = armature
            armature.select_set(True)

        # check if need to create bone
        bone = context.active_bone

        if bone is None or bone.name.startswith("ik"):
            return {"CANCELLED"}
        
        for other_bone in armature.pose.bones:
            if other_bone.name == "ik_" + bone.name:
                return {"CANCELLED"}

        bone_name = context.active_bone.name

        # create new ik bone
        bpy.ops.object.mode_set(mode="EDIT")
        ik_bone = armature.data.edit_bones.new("ik_" + bone_name)
        root = armature.data.edit_bones["root"]
        ik_bone.head = (0.0, 0.0, 0.0)
        ik_bone.tail = (0.0, 0.0, 0.1)
        ik_bone.parent = armature.data.edit_bones["root"]
        ik_bone_name = ik_bone.name

        bpy.ops.object.mode_set(mode="POSE")
        ik_bone = armature.pose.bones[ik_bone_name]
        ik_bone.custom_shape = bpy.data.objects["ik_bone_ref"]
        ik_bone.use_custom_shape_bone_size = False

        bone = armature.pose.bones[bone_name]
        ik_bone_start = bone.matrix @ bone.location

        # check if group for ik bones already exist
        is_group_exist = False
        for group in armature.pose.bone_groups:
            if group.name == "ik_bones":
                is_group_exist = True
                ik_group = group

        # create group if not exist
        if not is_group_exist:
            ik_group = bpy.ops.pose.group_add()
            ik_group = armature.pose.bone_groups[-1]
            ik_group.name = "ik_bones"
            ik_group.color_set = "THEME01"

        ik_bone.bone_group = ik_group

        # set positon of ik bone
        bpy.ops.object.mode_set(mode="EDIT")
        ik_bone = armature.data.edit_bones[ik_bone_name]
        ik_bone.translate(ik_bone_start)
        bpy.ops.object.mode_set(mode="POSE")

        return {'FINISHED'}

class ObjectVPoserRemoveIKBone(bpy.types.Operator):
    """Remove current ik bone"""
    bl_idname = "object.vposer_remove_ik_bone"
    bl_label = "Remove current ik bone"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            if context.active_object.mode == 'POSE':
                return True
            else: 
                return False
        except: return False


    def execute(self, context):
        armature = context.object

        if not check_compatibility(self, armature):
            return {"CANCELLED"}

        bone = context.active_bone

        if bone is not None and bone.name.startswith("ik"):
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.armature.delete()
            bpy.ops.object.mode_set(mode="POSE")
        else:
            return {"CANCELLED"}

        return {'FINISHED'}

class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm: BodyModel,
                 n_joints: int=22,
                 ):
        super(SourceKeyPoints, self).__init__()
        self.bm = bm
        self.n_joints = n_joints

    # compute joint locations of given body parameters
    def forward(self, body_parms):
        new_body = self.bm(**body_parms)

        return {'source_kpts':new_body.Jtr[:,:self.n_joints]}

# compute mse loss only for ik bones
class MaskedMSELoss(torch.nn.MSELoss):
    def __init__(self, mask):
        self.mask = mask
        super(MaskedMSELoss, self).__init__(reduction='sum')

    def forward(self, input, target):
        loss = super(MaskedMSELoss, self).forward(input[0][self.mask], 
                                                  target[0][self.mask])
        return loss 

class ObjectVPoserIK(bpy.types.Operator):
    """Optimize position to fit IK bones"""
    bl_idname = "object.vposer_ik"
    bl_label = "Optimize position to fit IK bones"
    bl_options = {'REGISTER', 'UNDO'}


    @classmethod
    def poll(cls, context):
        try:
            if context.active_object.mode == 'POSE':
                return True
            else: 
                return False
        except: return False

    def execute(self, context):
        armature = context.object

        if not check_compatibility(self, armature):
            return {"CANCELLED"}

        # get ik bones and their position
        ik_bones = {}

        for bone in armature.pose.bones:
            bone_group = bone.bone_group

            if bone_group is not None and bone_group.name == "ik_bones":
                ik_bones[bone.name[3:]] = bone.head

        if len(ik_bones) == 0:
            return {"CANCELLED"}

        smplx = sys.modules['smplx_blender_addon']

        # get target points positions and mask for mse loss calculation
        body_bones_names, body_pose = get_body_parameters(smplx, armature)
        root_name = smplx.SMPLX_JOINT_NAMES[0]
        target_pts = torch.zeros((1, len(body_bones_names) + 1, 3),
                dtype=torch.float)
        mask = torch.zeros(len(body_bones_names) + 1).type(torch.bool)

        for i, bone_name in enumerate([root_name] + body_bones_names):
            if bone_name in ik_bones:
                bone_location = ik_bones[bone_name]
                mask[i] = True
            else:
                bone = armature.pose.bones[bone_name]
                bone_location = (0, 0, 0)

            target_pts[0][i] = torch.Tensor((bone_location[0], 
                                             bone_location[2], 
                                             -bone_location[1]))
        
        # initiate ik engine
        data_loss = MaskedMSELoss(mask)
        stepwise_weights = [{'data': 10., 'poZ_body': .01, 'betas': .5},]
        optimizer_args = {'type':'LBFGS', 'max_iter':30, 'lr': 1, 
                          'tolerance_change': 1e-4, 'history_size':200}

        ik_engine = src.ik_engine.IK_Engine(
            vp_model=VP,
            data_loss=data_loss,
            stepwise_weights=stepwise_weights,
            optimizer_args=optimizer_args
        )

        # compute new body parameters from initial position by ik engine
        if "female" in armature:
            bm = FEMALE_MODEL
        elif "male" in armature:
            bm = MALE_MODEL
        else:
            bm = NEUTRAL_MODEL

        source_pts = SourceKeyPoints(bm=bm)
        root_orient = smplx.rodrigues_from_pose(armature, root_name)
        root_orient = \
            torch.Tensor(root_orient).reshape((1,-1)).type(torch.float)
        ik_res = ik_engine(
            source_pts, 
            target_pts, 
            {"pose_body": body_pose,"root_orient" : root_orient}
        )

        # apply new body parameters
        body_pose_rec = ik_res["pose_body"]
        for i, bone_name in enumerate(body_bones_names):
            smplx.set_pose_from_rodrigues(armature, bone_name, 
                                          body_pose_rec[0][3*i : 3*(i+1)])

        root_orient_rec = ik_res["root_orient"]
        smplx.set_pose_from_rodrigues(armature, root_name, root_orient_rec[0])

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

        col.separator()
        col.label(text="Inverse kinematics:")

        row = col.row(align=True)
        row.operator("object.vposer_add_ik_bone", text="Add IK bone")
        row.operator("object.vposer_remove_ik_bone", text="Remove IK bone")
        col.operator("object.vposer_ik", text="IK")

def register():
    bpy.utils.register_class(ObjectVPoserUpdatePose)
    bpy.utils.register_class(ObjectVPoserGeneratePose)
    bpy.utils.register_class(ObjectVPoserIK)
    bpy.utils.register_class(ObjectVPoserAddIKBone)
    bpy.utils.register_class(ObjectVPoserRemoveIKBone)
    bpy.utils.register_class(SMPLX_PT_VPoser)

def unregister():
    bpy.utils.unregister_class(ObjectVPoserPoseUpdate)
    bpy.utils.unregister_class(ObjectVPoserGeneratePose)
    bpy.utils.unregister_class(ObjectVPoserIK)
    bpy.utils.unregister_class(ObjectVPoserAddIKBone)
    bpy.utils.unregister_class(ObjectVPoserRemoveIKBone)
    bpy.utils.unregister_class(SMPLX_PT_VPoser)

if __name__ == "__main__":
    register()
