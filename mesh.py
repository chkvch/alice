import numpy as np
class mesh:

    def __init__(self, nz, params={}):

        # default mesh
        self.mesh_params = {
            'mesh_func_type':'flat_with_surface_exponential_core_gaussian',
            # 'width_transition_mesh_boost':6e-2,
            # 'amplitude_transition_mesh_boost':0.1,
            'amplitude_surface_mesh_boost':3e5,
            'width_surface_mesh_boost':1e-2,
            'amplitude_core_mesh_boost':1e5,
            'width_core_mesh_boost':1e-2, # 3e-1, # 1e-1
            'fmean_core_bdy_mesh_boost':0e0 # 1e-1 # 3e-1
            }
        # overwrite with any passed by user
        for key, value in params.items():
            self.mesh_params[key] = value

        # t = np.linspace(0, 1, nz)
        # assumes t runs from 0 at center to 1 at surface
        f0 = np.linspace(0, 1, nz)
        density_f0 = 1. / np.diff(f0)
        density_f0 = np.insert(density_f0, 0, density_f0[0])
        norm = np.mean(density_f0)
        density_f0 += self.mesh_params['amplitude_surface_mesh_boost'] * f0 * np.exp((f0 - 1.) / self.mesh_params['width_surface_mesh_boost']) * norm
        # density_f0 += self.mesh_params['amplitude_core_mesh_boost'] * np.exp(-(f0 - self.mesh_params['fmean_core_bdy_mesh_boost']) ** 2 / 2. / self.mesh_params['width_core_mesh_boost'] ** 2) * norm
        density_f0 += self.mesh_params['amplitude_core_mesh_boost'] * np.exp(-f0 / self.mesh_params['width_surface_mesh_boost']) * norm
        out = np.cumsum(1. / density_f0)
        out -= out[0]
        out /= out[-1]
        self.mesh_func = out
