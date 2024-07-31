from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class FrankaSoftclawView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "FrankaView",
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        self._claw_fixed_body = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/qbsoftclaw_body", name="_claw_fixed_body_view", reset_xform_properties=False, track_contact_forces=True)

        self._claw_moving_body = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/qbsoftclaw/qbsoftclaw_shaft_link", name="_claw_moving_body_view", reset_xform_properties=False, track_contact_forces=True)

        self._caw_fixed_finger = RigidPrimView(
            prim_paths_expr="/World/envs/.*/franka/fixed_finger_pad",
            name="_caw_fixed_finger_view",
            reset_xform_properties=False,
            track_contact_forces=True
        )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self._gripper_indices = [self.get_dof_index("qbsoftclaw_shaft_joint")]

    @property
    def gripper_indices(self):
        return self._gripper_indices
