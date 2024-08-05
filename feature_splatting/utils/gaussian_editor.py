import gc
import threading
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from feature_splatting.utils.mpm_engine.mpm_solver import MPMSolver

class gaussian_editor:
    # TODO(roger): this should be integrated with viewer_utils
    def __init__(self):
        self.meta_editing_dict = {}
        self.particle_modification_buffer = {}
        # NS is asynchonous and the pre-process/post-process functions may be called
        # in different threads

        # There may be a better way to do this, but for now we use a lock
        self.editors_lock = threading.Lock()

    def register_object_minimax(self, xyz_min, xyz_max):
        # Object bounding box minimax
        self.meta_editing_dict['xyz_min'] = torch.tensor(xyz_min).cuda().float()
        self.meta_editing_dict['xyz_max'] = torch.tensor(xyz_max).cuda().float()
    
    def register_ground_transform(self, ground_R, ground_T):
        self.meta_editing_dict['ground_R_np'] = ground_R.copy()
        self.meta_editing_dict['ground_T_np'] = ground_T.copy()
        self.meta_editing_dict['ground_R'] = torch.tensor(ground_R).cuda().float()
        self.meta_editing_dict['ground_T'] = torch.tensor(ground_T).cuda().float()
        up_gravity_vec = np.array((0, 1, 0))
        up_gravity_vec = ground_R.T @ up_gravity_vec
        self.meta_editing_dict['up_gravity_vec_np'] = up_gravity_vec

    def prepare_editing_dict(self, translation_vec, yaw_deg, physics_sim_flag):
        ret_editing_dict = {}
        if 'ground_R' in self.meta_editing_dict and 'ground_T' in self.meta_editing_dict:
            # Ground-aligned translation
            trans_vec_np = np.array(translation_vec)
            if trans_vec_np.any():
                trans_vec_gpu = torch.tensor(trans_vec_np).float().cuda()
                ret_editing_dict["translation"] = self.meta_editing_dict['ground_R'].T @ trans_vec_gpu
            # Ground-aligned rotation
            if yaw_deg:
                # TODO(roger): currently only support yaw rotation around gravity axis
                rot_axis = self.meta_editing_dict['up_gravity_vec_np'] / np.linalg.norm(self.meta_editing_dict['up_gravity_vec_np'])
                r = Rotation.from_rotvec(yaw_deg * rot_axis, degrees=True)
                rot_mat = r.as_matrix()
                ret_editing_dict["rotation"] = torch.tensor(rot_mat).float().cuda()
            if physics_sim_flag:
                if "physics_sim" not in self.meta_editing_dict:
                    print("Initializing physics simulation engine...")
                    self.initialize_mpm_engine()
                ret_editing_dict["physics_sim"] = True
        if not physics_sim_flag:
            if "physics_sim" in self.meta_editing_dict:
                del self.meta_editing_dict["physics_sim"]
                gc.collect()
            
        return ret_editing_dict

    @torch.no_grad()
    def initialize_mpm_engine(self, youngs_modulus_scale=1, poisson_ratio=0.2):
        import taichi as ti
        ti.init(arch=ti.cuda, device_memory_GB=4.0)

        gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

        mpm = MPMSolver(res=(32, 32, 32), size=1, max_num_particles=2 ** 21,
                        E_scale=youngs_modulus_scale, poisson_ratio=poisson_ratio)

        self.meta_editing_dict["physics_sim"] = {
            "mpm": mpm,
            "gui": gui,
            "initialized": False
        }
    
    @torch.no_grad()
    def initialize_mpm_w_particles(self, init_particles_positions, infilling_downsample_ratio=0.2, ground_level=0.05, gravity=4):
        assert not self.meta_editing_dict["physics_sim"]["initialized"]
        real_gaussian_particle = init_particles_positions
        real_obj_center = real_gaussian_particle.mean(axis=0)

        # Add pseudo points from center of the object to all particles
        support_per_particles = 10

        support_particles_list = []

        for particles_idx in range(real_gaussian_particle.shape[0]):
            start_pos = real_obj_center
            end_pos = real_gaussian_particle[particles_idx]
            for support_idx in range(support_per_particles):
                # interpolate
                pos = (start_pos * (support_per_particles - support_idx) + end_pos * support_idx) / support_per_particles
                support_particles_list.append(pos)

        support_particles = np.array(support_particles_list)
        support_particles = np.random.permutation(support_particles)[:int(len(support_particles) * infilling_downsample_ratio)]

        all_particles = np.concatenate([real_gaussian_particle, support_particles], axis=0)

        # Align to ground
        particles = all_particles @ self.meta_editing_dict['ground_R_np'].T
        particles += self.meta_editing_dict['ground_T_np']

        # Normalize everything to a unit world box; x-z coordinates are centered at 0.5
        particle_max = particles.max(axis=0)
        particle_min = particles.min(axis=0)
        particle_min[1] = min(particle_min[1], ground_level)

        longest_side = max(particle_max - particle_min)

        particles[:, 0] /= longest_side
        particles[:, 1] /= longest_side
        particles[:, 2] /= longest_side

        # Align centers of x and z to 0.5 and set the bottom of the object to 0
        shift_constant = np.array([
            -particles[:,0].mean() + 0.5,
            -particles[:,1].min(),
            -particles[:,2].mean() + 0.5
        ])

        particles += shift_constant

        self.meta_editing_dict["physics_sim"]["mpm"].add_particles(particles=particles,
                    material=MPMSolver.material_sand,
                    color=0xFFFF00)

        self.meta_editing_dict["physics_sim"]["mpm"].add_surface_collider(point=(0.0, ground_level, 0.0),
                                    normal=(0, 1, 0),
                                    surface=self.meta_editing_dict["physics_sim"]["mpm"].surface_sticky)

        self.meta_editing_dict["physics_sim"]["mpm"].set_gravity((0, -gravity, 0))

        # Memorize constants
        self.meta_editing_dict["physics_sim"]["longest_side"] = longest_side
        self.meta_editing_dict["physics_sim"]["shift_constant"] = shift_constant
        self.meta_editing_dict["physics_sim"]["real_gaussian_particle_size"] = real_gaussian_particle.shape[0]
    
    @torch.no_grad()
    def physics_sim_step(self, timestep=4e-3):
        real_gaussian_particle_size = self.meta_editing_dict["physics_sim"]["real_gaussian_particle_size"]
        longest_side = self.meta_editing_dict["physics_sim"]["longest_side"]
        shift_constant = self.meta_editing_dict["physics_sim"]["shift_constant"]

        particles_info = self.meta_editing_dict["physics_sim"]["mpm"].particle_info()

        real_gaussian_pos = particles_info['position'][:real_gaussian_particle_size]

        ret_trajectory = real_gaussian_pos.copy()

        self.meta_editing_dict["physics_sim"]["mpm"].step(timestep)

        ret_trajectory -= shift_constant
        ret_trajectory *= longest_side

        # Reverse rigid transformation
        ret_trajectory = ret_trajectory - self.meta_editing_dict['ground_T_np']
        ret_trajectory = ret_trajectory @ self.meta_editing_dict['ground_R_np']

        return ret_trajectory

    @torch.no_grad()
    def pre_rendering_process(self, means, opacities, scales, quats, editing_dict, view_main_obj_only=False, **kwargs):
        self.editors_lock.acquire()
        assert not self.particle_modification_buffer, "Particle modification buffer is not empty"
        # If object bbox is set and view_main_obj_only is set, hide particles outside the bbox
        if 'xyz_min' in self.meta_editing_dict:
            assert 'min_offset' in kwargs and 'max_offset' in kwargs
            # Get object bounding box
            bbox_particle_idx = self.filter_particles_ground_bbox(means,
                                                                    kwargs['min_offset'],
                                                                    kwargs['max_offset'])

            # Hide particles outside the bounding box?
            if view_main_obj_only:
                bg_idx = ~bbox_particle_idx
                if 'original_opacities' not in self.particle_modification_buffer:
                    self.particle_modification_buffer['original_opacities'] = opacities.clone()
                opacities[bg_idx] = -5 # in-place modification
            
            if "translation" in editing_dict:
                if 'original_means' not in self.particle_modification_buffer:
                    self.particle_modification_buffer["original_means"] = means.clone()
                means[bbox_particle_idx] = means[bbox_particle_idx] + editing_dict["translation"]
            
            if "rotation" in editing_dict:
                if 'original_means' not in self.particle_modification_buffer:
                    self.particle_modification_buffer["original_means"] = means.clone()
                if 'original_quats' not in self.particle_modification_buffer:
                    self.particle_modification_buffer["original_quats"] = quats.clone()
                
                rot_mat = editing_dict["rotation"]

                # Rotate x/y/z
                selected_pts = means[bbox_particle_idx]
                object_center = selected_pts.mean(dim=0)

                selected_pts = selected_pts - object_center
                selected_pts = rot_mat @ selected_pts.T
                selected_pts = selected_pts.T
                selected_pts = selected_pts + object_center

                means[bbox_particle_idx] = selected_pts

                # Rotate covariance
                r = quats[bbox_particle_idx]
                rot_mat = rot_mat.reshape((1, 3, 3))  # (N, 3, 3)

                r = get_gaussian_rotation(rot_mat.cpu().numpy(), r)

                quats[bbox_particle_idx] = r
            
            if "physics_sim" in editing_dict and editing_dict["physics_sim"]:
                if 'original_means' not in self.particle_modification_buffer:
                    self.particle_modification_buffer["original_means"] = means.clone()
                if not self.meta_editing_dict["physics_sim"]["initialized"]:
                    selected_particles = means[bbox_particle_idx]
                    self.initialize_mpm_w_particles(selected_particles.cpu().numpy())
                    self.meta_editing_dict["physics_sim"]["initialized"] = True
                import time
                start_cp = time.time()
                particle_positions_np = self.physics_sim_step()
                print("Physics sim step time: ", time.time() - start_cp)
                assert particle_positions_np.shape[0] == bbox_particle_idx.sum()
                means[bbox_particle_idx] = torch.tensor(particle_positions_np).cuda().float()
            
            # TODO(roger): implement scaling? Below are scaling editing code for INRIA impl
            # gaussians._xyz[selected_obj_idx] = gaussians._xyz[selected_obj_idx] / scale
            # gaussians._scaling = gaussians.inverse_opacity_activation(
            #     gaussians.scaling_activation(gaussians._scaling[selected_obj_idx]) / scale
            # )

    @torch.no_grad()
    def post_rendering_process(self, means, opacities, quats, scales):
        """Inverse function of pre_rendering_process, which reverses
        the transformation applied to the particles.
        """
        if 'original_opacities' in self.particle_modification_buffer:
            opacities.copy_(self.particle_modification_buffer['original_opacities'])
            del self.particle_modification_buffer['original_opacities']
        if 'original_means' in self.particle_modification_buffer:
            means.copy_(self.particle_modification_buffer['original_means'])
            del self.particle_modification_buffer['original_means']
        if 'original_quats' in self.particle_modification_buffer:
            quats.copy_(self.particle_modification_buffer['original_quats'])
            del self.particle_modification_buffer['original_quats']
        if 'original_scales' in self.particle_modification_buffer:
            scales.copy_(self.particle_modification_buffer['original_scales'])
            del self.particle_modification_buffer['original_scales']
        self.editors_lock.release()

    def filter_particles_ground_bbox(self, means, min_offset, max_offset):
        ground_R = self.meta_editing_dict['ground_R']
        ground_T = self.meta_editing_dict['ground_T']
        particles = means @ ground_R.T
        particles += ground_T
        xyz_min = self.meta_editing_dict['xyz_min'] - min_offset
        xyz_max = self.meta_editing_dict['xyz_max'] + max_offset
        bbox_particles_idx = ((particles > xyz_min) & (particles < xyz_max)).all(dim=1)
        return bbox_particles_idx

def get_gaussian_rotation(rot_mat, r):
    # Rotate unnormalized quaternion by rotation matrix, and gives back unnormalized quats
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    R = rot_mat @ R.detach().cpu().numpy()

    # Convert back to quaternion
    r = Rotation.from_matrix(R).as_quat()
    r[:, [0, 1, 2, 3]] = r[:, [3, 0, 1, 2]]  # x,y,z,w -> r,x,y,z
    r = torch.from_numpy(r).cuda().float()

    r = r * norm[:, None]
    return r

