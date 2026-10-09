import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

@dataclass
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: torch.Tensor   # [3]
    origin_rpy: torch.Tensor   # [3]
    axis: torch.Tensor         # [3]

def rpy_to_matrix(roll: float, pitch: float, yaw: float, device=None, dtype=torch.float32):
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    R = torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ], dtype=dtype, device=device)
    return R

def make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    R: [..., 3, 3]
    t: [..., 3]
    return: [..., 4, 4]
    """
    batch_shape = R.shape[:-2]
    T = torch.zeros(*batch_shape, 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T

def transform_inverse(T: torch.Tensor) -> torch.Tensor:
    """
    T: [..., 4, 4]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)
    return make_transform(R_inv, t_inv)

def transform_mul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)

def build_joint_pos_dict(dof_names, dof_pos):
    """
    dof_pos: [N, num_dofs]
    """
    joint_pos_dict = {}
    for i, name in enumerate(dof_names):
        joint_pos_dict[name] = dof_pos[:, i]
    return joint_pos_dict

def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    axis: [3] or [N,3], should be unit axis
    angle: [] or [N]
    return: [3,3] or [N,3,3]
    """
    if axis.ndim == 1:
        axis = axis.unsqueeze(0)
        angle = angle.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]

    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1.0 - c

    R = torch.zeros(axis.shape[0], 3, 3, dtype=axis.dtype, device=axis.device)
    R[:, 0, 0] = c + x * x * C
    R[:, 0, 1] = x * y * C - z * s
    R[:, 0, 2] = x * z * C + y * s

    R[:, 1, 0] = y * x * C + z * s
    R[:, 1, 1] = c + y * y * C
    R[:, 1, 2] = y * z * C - x * s

    R[:, 2, 0] = z * x * C - y * s
    R[:, 2, 1] = z * y * C + x * s
    R[:, 2, 2] = c + z * z * C

    if squeeze_back:
        return R[0]
    return R


class SimpleURDFKinematics:
    def __init__(self, urdf_path: str, device="cpu", dtype=torch.float32):
        self.urdf_path = urdf_path
        self.device = torch.device(device)
        self.dtype = dtype

        self.joints: Dict[str, JointInfo] = {}
        self.child_to_joint: Dict[str, JointInfo] = {}
        self.parent_to_children: Dict[str, List[str]] = {}
        self.links: set = set()

        self._parse_urdf()

    def _parse_urdf(self):
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()

        for link in root.findall("link"):
            self.links.add(link.attrib["name"])

        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            joint_type = joint.attrib["type"]

            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]

            origin = joint.find("origin")
            if origin is not None:
                xyz_str = origin.attrib.get("xyz", "0 0 0")
                rpy_str = origin.attrib.get("rpy", "0 0 0")
            else:
                xyz_str = "0 0 0"
                rpy_str = "0 0 0"

            axis_elem = joint.find("axis")
            if axis_elem is not None:
                axis_str = axis_elem.attrib.get("xyz", "0 0 1")
            else:
                axis_str = "0 0 1"

            origin_xyz = torch.tensor(
                [float(x) for x in xyz_str.split()],
                dtype=self.dtype, device=self.device
            )
            origin_rpy = torch.tensor(
                [float(x) for x in rpy_str.split()],
                dtype=self.dtype, device=self.device
            )
            axis = torch.tensor(
                [float(x) for x in axis_str.split()],
                dtype=self.dtype, device=self.device
            )

            axis_norm = torch.norm(axis)
            if axis_norm > 1e-8:
                axis = axis / axis_norm

            info = JointInfo(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
            )

            self.joints[name] = info
            self.child_to_joint[child] = info
            self.parent_to_children.setdefault(parent, []).append(child)

    def get_chain(self, root_link: str, tip_link: str) -> List[JointInfo]:
        """
        Return ordered joints from root_link -> tip_link
        """
        chain = []
        current = tip_link

        while current != root_link:
            if current not in self.child_to_joint:
                raise ValueError(f"Cannot find joint leading to child link {current}. "
                                 f"Check if root_link='{root_link}' is correct.")
            joint = self.child_to_joint[current]
            chain.append(joint)
            current = joint.parent

        chain.reverse()
        return chain
    
    def joint_transform(self, joint: JointInfo, q: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        """
        joint: JointInfo
        q: [N] for movable joint, or None for fixed joint
        return: [N, 4, 4]
        """
        R0 = rpy_to_matrix(
            float(joint.origin_rpy[0]),
            float(joint.origin_rpy[1]),
            float(joint.origin_rpy[2]),
            device=self.device,
            dtype=self.dtype
        )
        t0 = joint.origin_xyz
        T0 = make_transform(
            R0.unsqueeze(0).repeat(batch_size, 1, 1),
            t0.unsqueeze(0).repeat(batch_size, 1)
        )

        if joint.joint_type == "fixed":
            return T0

        if q is None:
            raise ValueError(f"Joint {joint.name} requires q but got None.")

        if joint.joint_type in ["revolute", "continuous"]:
            axis = joint.axis.unsqueeze(0).repeat(batch_size, 1)
            Rm = axis_angle_to_matrix(axis, q)
            tm = torch.zeros(batch_size, 3, dtype=self.dtype, device=self.device)
            Tm = make_transform(Rm, tm)
            return transform_mul(T0, Tm)

        elif joint.joint_type == "prismatic":
            Rm = torch.eye(3, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
            tm = joint.axis.unsqueeze(0) * q.unsqueeze(-1)
            Tm = make_transform(Rm, tm)
            return transform_mul(T0, Tm)

        else:
            raise NotImplementedError(f"Unsupported joint type: {joint.joint_type}")
        
    def fk_link_pose(
        self,
        root_link: str,
        tip_link: str,
        joint_pos_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        root_link -> tip_link FK
        return: [N, 4, 4], pose of tip_link expressed in root_link frame
        """
        chain = self.get_chain(root_link, tip_link)

        # infer batch size
        batch_size = None
        for _, q in joint_pos_dict.items():
            batch_size = q.shape[0]
            break
        if batch_size is None:
            raise ValueError("joint_pos_dict is empty.")

        T = torch.eye(4, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)

        for joint in chain:
            if joint.joint_type == "fixed":
                T_joint = self.joint_transform(joint, None, batch_size)
            else:
                if joint.name not in joint_pos_dict:
                    raise KeyError(f"Missing joint angle for joint {joint.name}")
                q = joint_pos_dict[joint.name]
                T_joint = self.joint_transform(joint, q, batch_size)

            T = transform_mul(T, T_joint)

        return T
    
class PelvisEstimatorFromFeetMid:
    def __init__(
        self,
        urdf_path: str,
        pelvis_link: str,
        left_foot_link: str,
        right_foot_link: str,
        device="cpu",
        dtype=torch.float32
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.kin = SimpleURDFKinematics(urdf_path, device=device, dtype=dtype)

        self.pelvis_link = pelvis_link
        self.left_foot_link = left_foot_link
        self.right_foot_link = right_foot_link


    def feet_pose_in_pelvis(
        self,
        joint_pos_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return:
            T_PL: [N,4,4] left foot pose in pelvis frame
            T_PR: [N,4,4] right foot pose in pelvis frame
        """
        T_PL = self.kin.fk_link_pose(self.pelvis_link, self.left_foot_link, joint_pos_dict)
        T_PR = self.kin.fk_link_pose(self.pelvis_link, self.right_foot_link, joint_pos_dict)
        return T_PL, T_PR
    
    def feet_mid_frame_in_pelvis(
    self,
    joint_pos_dict: Dict[str, torch.Tensor],
    up_axis_in_pelvis: torch.Tensor = None,
) -> torch.Tensor:
        T_PL, T_PR = self.feet_pose_in_pelvis(joint_pos_dict)

        pL = T_PL[:, :3, 3]
        pR = T_PR[:, :3, 3]
        pM = 0.5 * (pL + pR)

        if up_axis_in_pelvis is None:
            batch_size = pL.shape[0]
            up_axis_in_pelvis = torch.zeros(batch_size, 3, device=self.device, dtype=self.dtype)
            up_axis_in_pelvis[:, 2] = 1.0
#         up_axis_in_pelvis = up_axis_in_pelvis / torch.norm(
#     up_axis_in_pelvis, dim=-1, keepdim=True
# ).clamp(min=1e-8)
        z_axis = up_axis_in_pelvis
        z_axis = z_axis / torch.norm(z_axis, dim=-1, keepdim=True).clamp(min=1e-8)

        y_axis = pL - pR
        y_axis = y_axis - (y_axis * z_axis).sum(dim=-1, keepdim=True) * z_axis
        y_axis = y_axis / torch.norm(y_axis, dim=-1, keepdim=True).clamp(min=1e-8)

        x_axis = torch.cross(y_axis, z_axis, dim=-1)
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True).clamp(min=1e-8)

        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis / torch.norm(y_axis, dim=-1, keepdim=True).clamp(min=1e-8)

        R_PM = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        T_PM = make_transform(R_PM, pM)
        return T_PM
    

    def pelvis_position_in_midfeet(
    self,
    joint_pos_dict: Dict[str, torch.Tensor],
    up_axis_in_pelvis: torch.Tensor = None,
) -> torch.Tensor:
        T_PM = self.feet_mid_frame_in_pelvis(joint_pos_dict, up_axis_in_pelvis)
        T_MP = transform_inverse(T_PM)
        return T_MP[:,:3,3]
    
