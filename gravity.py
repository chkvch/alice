# a fourth-order theory of figures based on
# the summmary in appendix B of Nettelmann (2017)
# https://arxiv.org/abs/1708.06177v1
# 2017arXiv170806177N

import numpy as np
import const
from scipy.optimize import root
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import splrep, splev, interp1d
from scipy.special import legendre
import time
import os
import ongp

class ConvergenceError(Exception):
    pass

class ToFAdjustError(Exception):
    pass
    
class model_container:
    '''
    container for quantities not fundamental to the tof4 model, e.g., composition, etc.
    '''
    pass    

class tof4:
    def __init__(self, tof_params={}, mesh_params={}):

        # parse contents of the params dictionary
        self.max_iters_outer = tof_params['max_iters_outer'] if 'max_iters_outer' in tof_params else 50
        self.max_iters_inner = tof_params['max_iters_inner'] if 'max_iters_inner' in tof_params else 2
        self.verbosity = tof_params['verbosity'] if 'verbosity' in tof_params else 1

        # tolerance for relative change in all the j2n
        self.j2n_rtol = tof_params['j2n_rtol'] if 'j2n_rtol' in tof_params else 1e-4
        # tolerance for relative error in the total mass
        self.mtot_rtol = tof_params['mtot_rtol'] if 'mtot_rtol' in tof_params else 1e-5
        # tolerance for relative error in the mean He mass fraction
        self.ymean_rtol = tof_params['ymean_rtol'] if 'ymean_rtol' in tof_params else 1e-3 
        # number of zones in the model; use thousands if fidelity in j2n is important
        self.nz = tof_params['nz'] if 'nz' in tof_params else 512

        if not 'save_vector_output' in tof_params.keys():
            tof_params['save_vector_output'] = True

        self.params = tof_params
        self.mesh_params = mesh_params

        if 'output_path' in tof_params.keys():
            output_path = tof_params['output_path']
        else:
            output_path = 'output'
        if 'uid' in tof_params.keys():
            self.uid = tof_params['uid']
            self.output_path = '{}/{:n}'.format(output_path, self.uid) # one run goes into a unique subfolder within params['output_path']
        else:
            self.uid = int(1e6*time.time())
            self.output_path = '{}/{:n}'.format(output_path, self.uid) # one run goes into a unique subfolder within params['output_path']
            while os.path.exists(self.output_path):
                # uid is apparently not unique in output_path, increment it until unique
                self.uid += 1
                self.output_path = '{}/{:n}'.format(output_path, self.uid) # one run goes into a unique subfolder within params['output_path']

        assert type(self.uid) is int, 'assigned a unique identifier that is not an integer'
        while True: # not clear why this is not caught just above
            try:
                os.makedirs(self.output_path)
                break
            except FileExistsError: # increment uid slowly until it's new
                self.uid += 1
                self.output_path = '{}/{}'.format(output_path, self.uid) # one run goes into a unique subfolder within params['output_path']

        if 'method_for_aa2n_solve' in tof_params.keys():
            self.method_for_aa2n_solve = tof_params['method_for_aa2n_solve']
        else:
            self.method_for_aa2n_solve = 'cubic 32'

    def initialize_model(self):

        self.small = self.params['small'] # the smallness parameter, m
        self.small_start = self.small
        assert self.small >= 0., 'small must be non-negative.'
        assert self.small < 1., 'no supercritical rotation fam.'
        if self.small == 0. and self.verbosity > 0: print('warning: rotation is off.')

        self.mtot = self.params['mtot']
        self.rtot = self.params['req'] * (1. - self.small ** 2)

        # use ongp to generate a basic initial model. this only needs to be a starting point for
        # rho, p, r; details like the Y and Z distributions will be enforced later during ToF
        # iterations in the adjust_* method appropriate to the chosen model_type.
        # set some params that ongp.evol needs to instantiate an ongp.evol object
        ongp_evol_params = {}
        ongp_evol_params['nz'] = self.nz
        ongp_evol_params['atm_option'] = self.params['atm_option']
        ongp_evol_params['atm_planet'] = self.params['atm_planet']
        ongp_evol_params['radius_rtol'] = 1e-2 # need not be a good model at all
        ongp_evol_params['hhe_eos_option'] = self.params['hhe_eos_option']
        ongp_evol_params['z_eos_option'] = self.params['z_eos_option']
        ongp_evol_params['f_ice'] = self.params['ice_to_rock']
        e = ongp.evol(ongp_evol_params, self.mesh_params)
        # set params that ongp.evol.static needs to build a specific static model.
        # composition and three-layer structure is arbitrary because it's just an initial guess
        ongp_model_params = {'mtot':self.mtot, 't1':self.params['t1'], 'z1':0.1, 'y1':0.27, 'model_type':'three_layer'}
        if 'mcore' in self.params: ongp_model_params['mcore'] = self.params['mcore']
        e.static(ongp_model_params)

        # check total mass off the bat
        dm = 4. / 3 * np.pi * e.rho[1:] * (e.r[1:] ** 3 - e.r[:-1] ** 3)
        dm = np.insert(dm, 0, 4. / 3 * np.pi * e.rho[0] * e.r[0] ** 3)
        m_calc = np.cumsum(dm)
        mtot_calc = m_calc[-1]
        # print( 'initial model: mtot_calc, reldiff', mtot_calc, abs(self.mtot - mtot_calc) / self.mtot)

        self.rho = e.rho
        self.p = e.p
        self.l = e.r # pretty arbitrary to copy the radius grid from the ongp model, but it works
        # ensure some zones very close to center; not necessary for many purposes
        n = 50
        self.l[:n] = np.linspace(1e-4, self.l[n-1], n)
        # self.l = np.linspace(1e-2, 1, len(e.r)) ** 2 * e.r[-1] # works, but poor resolution near surface
        # self.l = np.logspace(0, -2, len(e.r))[::-1] * e.r[-1] # doesn't work

        # copy over the important parts of this initial ongp.evol instance
        self.model = model_container()

        self.model.z1 = self.params['z1']
        if 'z2' in list(self.params): self.model.z2 = self.params['z2']
        if 'c' in list(self.params): self.model.c = self.params['c']
        self.model.t1 = self.params['t1']
        if 'mcore' in list(self.params): self.model.mcore = self.params['mcore']
        if 'mcore' in list(self.params): self.model.kcore = e.kcore
        self.model.nz = self.nz

        self.model.p = e.p
        self.model.t = e.t
        self.model.y = e.y
        self.model.z = e.z

        self.model.grada = e.grada
        self.model.gradt = e.gradt

        if 'pt' in list(self.params): self.model.ptrans = self.params['pt']

        self.model.hhe_eos = e.hhe_eos
        self.model.z_eos = e.z_eos
        if hasattr(e, 'z_eos_low_t'): self.model.z_eos_low_t = e.z_eos_low_t
        self.model.z_eos_option = self.params['z_eos_option']

        if self.params['model_type'] == 'three_layer':
            self.model.mcore = e.mcore
            self.model.y1 = self.params['y1']
        elif self.params['model_type'] == 'cosine_yz':
            self.model.c = self.params['c'] # y/z gradient centroid
            self.model.w = self.params['w'] # y/z gradient full-width
            self.model.y1_xy = self.params['y1_xy'] # y/(x+y) of uniform outer envelope
            try:
                self.model.y2_xy = self.params['y2_xy'] # y/(x+y) of "helium shell"
            except KeyError:
                self.model.y2_xy = None # adjust will catch this and set y2_xy=y1_xy
        elif self.params['model_type'] == 'linear_yz':
            self.model.c = self.params['c'] # y/z gradient centroid
            self.model.w = self.params['w'] # y/z gradient full-width
            self.model.y1_xy = self.params['y1_xy'] # y/(x+y) of uniform outer envelope
            try:
                self.model.y2_xy = self.params['y2_xy'] # y/(x+y) of "helium shell"
            except KeyError:
                self.model.y2_xy = None # adjust will catch this and set y2_xy=y1_xy

        self.model.get_rho_xyz = e.get_rho_xyz
        self.model.get_rho_z = e.get_rho_z

    def initialize_tof_vectors(self):
        if self.l[0] == 0.:
            self.l[0] = self.l[1] / 2
        self.rm = self.l[-1]

        self.nz = len(self.rho)
        # r_pol :: the polar radii
        self.r_pol = np.zeros(self.nz)
        # r_eq :: equatorial radii
        self.r_eq = np.zeros(self.nz)
        # s_2n :: the figure functions
        self.s0 = np.zeros(self.nz)
        self.s2 = np.zeros(self.nz)
        self.s4 = np.zeros(self.nz)
        self.s6 = np.zeros(self.nz)
        self.s8 = np.zeros(self.nz)
        # A_2n
        self.aa0 = np.zeros(self.nz)
        self.aa2 = np.zeros(self.nz)
        self.aa4 = np.zeros(self.nz)
        self.aa6 = np.zeros(self.nz)
        self.aa8 = np.zeros(self.nz)
        # S_2n
        self.ss0 = np.zeros(self.nz)
        self.ss2 = np.zeros(self.nz)
        self.ss4 = np.zeros(self.nz)
        self.ss6 = np.zeros(self.nz)
        self.ss8 = np.zeros(self.nz)
        # S_2n^'
        self.ss0p = np.zeros(self.nz)
        self.ss2p = np.zeros(self.nz)
        self.ss4p = np.zeros(self.nz)
        self.ss6p = np.zeros(self.nz)
        self.ss8p = np.zeros(self.nz)
        # set f0 (only needs to be done once)
        self.f0 = np.ones(self.nz)
        # J_2n
        self.j2 = 0.
        self.j4 = 0.
        self.j6 = 0.
        self.j8 = 0.
        self.j2n = np.array([self.j2, self.j4, self.j6, self.j8])

        # legendre polynomials for calculating radii from shape.
        # these provide scalar functions of mu := cos(theta).
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        self.pp0 = np.poly1d(legendre(0))
        self.pp2 = np.poly1d(legendre(2))
        self.pp4 = np.poly1d(legendre(4))
        self.pp6 = np.poly1d(legendre(6))
        self.pp8 = np.poly1d(legendre(8))


    def relax(self):

        color_by_outer_iteration = {0:'dodgerblue', 1:'gold', 2:'firebrick', 3:'forestgreen', 4:'purple', 5:'coral', 6:'teal'} # this takes me back

        import time
        time_start_outer = time.time()

        assert hasattr(self, 'mtot'), 'mtot not set; did you run self.initialize_model?'
        assert hasattr(self, 'rm'), 'rm not set; did you run self.initialize_tof_vectors?'

        self.rhobar = 3. * self.mtot / 4 / np.pi / self.rm ** 3 # this will be updated with mtot_calc

        self.outer_done = False
        self.j2n_last_outer = np.zeros(4)

        for self.outer_iteration in np.arange(self.max_iters_outer):

            if self.outer_iteration == self.max_iters_outer - 1:
                raise ConvergenceError('tof outer iteration failed to converge after %i iterations.' % self.max_iters_outer)

            self.inner_done = False
            time_start_inner = time.time()
            for self.inner_iteration in np.arange(self.max_iters_inner):

                # relax the shape for this rho(l)

                if 'use_radau_approximation' not in list(self.params) or not self.params['use_radau_approximation']:
                    # do full shape calculation
                    self.set_f2n_f2np() # polynomials of shape functions
                    self.set_ss2n_ss2np() # integrals over mass distribution weighted by f_2n, f_2n'
                    self.set_s2n() # numerical solve to get shape functions from A_2n = 0
                    self.set_req_rpol() # get new vectors of equatorial and polar radii from shape functions
                    self.set_j2n() # functions only of S_2n at surface and R_eq
                    self.req_radau = -1
                    self.j2_radau = -1

                    # if j2n are converged for this rho(l), we're done with inner iterations
                    if np.all(self.j2n != 0) and self.inner_iteration > 0:
                        if np.all(abs(self.dj2n) / self.j2n + 1e-14 < self.j2n_rtol):
                            if self.verbosity > 1:
                                data = self.inner_iteration, \
                                        self.j0, self.j2, self.j4, self.j6, self.j8, \
                                        self.r_eq[-1], self.rm, self.r_pol[-1], \
                                        self.q
                                print( ('%15i ' + ('%15.10f ' * 5) + ('%15g ' * 4)) % data)
                            if self.verbosity > 2:
                                print ('terminate inner loop; all dj2n/j2n < %g.' % self.j2n_rtol)
                            self.inner_done = True # but don't break yet, in case there's more output of interest
                else:
                    # early in outer iterations, can skip the full calculation and use Radau/Darwin
                    # approximation to the Clairaut ellipticity to quickly get in the right ballpark.
                    # on testing this doesn't help that much, mostly because after switching to the
                    # real calculation everything suddenly changes and it's almost back to square
                    # one in terms of the outer iterations converging. only saves maybe 200-300 ms.
                    self.set_f2n_f2np() # polynomials of shape functions
                    self.set_ss2n_ss2np() # integrals over mass distribution weighted by f_2n, f_2n' # 3-4 ms
                    self.set_j2n_radau()
                    self.r_eq = self.l * (1. + 0.5 * self.eps_radau) # equator -> p2(0) = 1/2 * (-1) = -1/2
                    self.r_pol = self.l * (1. - self.eps_radau) # pole -> p2(1) = 1/2 * (3 - 1) = 1
                    self.q = self.small * (self.r_eq[-1] / self.rm) ** 3
                    self.aa0 = np.ones_like(self.l)
                    self.inner_done = True

                if self.verbosity > 1:
                    # print various scalars
                    if self.inner_iteration == 0:
                        names = 'i_inner', "J0", "J2", "J4", "J6", "J8", 'R_eq', 'R_mean', 'R_pol', 'q'
                        print( '%15s ' * len(names) % names)
                    if not self.inner_done:
                        data = self.inner_iteration, \
                                self.j0, self.j2, self.j4, self.j6, self.j8, \
                                self.r_eq[-1], self.rm, self.r_pol[-1], \
                                self.q
                        print( ('%15i ' + ('%15.10f ' * 5) + ('%15g ' * 4)) % data)

                if self.inner_done or self.small == 0.:
                    break

            # we hit max inner iters although dj2n have not converged.
            # this is fine as long as it doesn't happen for late-stage outer iterations.
            if not self.inner_done and self.verbosity > 1:
                data = self.inner_iteration, \
                        self.j0, self.j2, self.j4, self.j6, self.j8, \
                        self.r_eq[-1], self.rm, self.r_pol[-1], \
                        self.small * (self.r_eq[-1] / self.rm) ** 3
                print (('%15i ' + ('%15.10f ' * 5) + ('%15g ' * 4)) % data)
                if self.small > 0. and self.verbosity > 2:
                    print ('warning: shape might not be fully converged.')

            self.et_inner_loop = time.time() - time_start_inner

            #
            # get the total potential from shape
            #
            self.set_aa0() # from the s_2n, S_2n, S_2n^'
            # NM recommends that for the sake of calculating U, for the sake of ease of late-stage
            # convergence, use target total mass rather than m_calc.
            self.rhobar_target = 3. * self.mtot / 4. / np.pi / self.rm ** 3
            self.u = -4. / 3 * np.pi * const.cgrav * self.rhobar_target * self.l ** 2 * self.aa0

            #
            # integrate hydrostatic balance and continuity
            #
            self.dl = np.diff(self.l)
            self.dl = np.insert(self.dl, 0, self.l[0])
            self.du = np.diff(self.u)

            if self.verbosity > 2: print( 'integrating hydro...')
            self.p[-1] = 1e6
            self.rho_mid = 0.5 * (self.rho[1:] + self.rho[:-1])
            for k in np.arange(self.nz)[::-1]:
                if k == self.nz - 1: continue
                assert self.rho[k+1] > 0., 'hydro integration found a bad density; cannot continue.'
                assert self.du[k] != 0., f'hydro integration found no change in potential at zone {k}; cannot continue.'
                self.p[k] = self.p[k+1] + self.rho_mid[k] * self.du[k]

            assert np.all(self.p > 0), 'bad pressure after integrate hydrostatic balance'
            self.model.p = self.p # self.model will need updated pressures for consistent rainout calculation

            # integrate continuity for current mass distribution
            dm = 4. / 3 * np.pi * self.rho[1:] * (self.l[1:] ** 3 - self.l[:-1] ** 3)
            self.model.dm = dm # he rainout calc in self.adjust will make use of this
            dm = np.insert(dm, 0, 4. / 3 * np.pi * self.rho[0] * self.l[0] ** 3)
            self.dm = dm
            self.m_calc = np.cumsum(dm)

            self.mtot_calc = self.m_calc[-1]
            if self.verbosity > 2:
                print( 'check continuity before touch density: mtot_calc, reldiff', self.mtot_calc, abs(self.mtot_calc - self.mtot) / self.mtot)

            # adjust mcore in underlying model, without reintegrating hydrostatic balance or continuity. only enforces the physical eos.
            if self.params['model_type'] == 'three_layer':
                if self.verbosity > 2:
                    print( 'tof will adjust core mass')
                # three layer model needs some massaging in correction applied to mcore to aid late-stage convergence
                if self.outer_iteration > 45:
                    f_mcore_correction = 1. / 27
                elif self.outer_iteration > 35:
                    f_mcore_correction = 1. / 18
                elif self.outer_iteration > 25:
                    f_mcore_correction = 1. / 9
                else:
                    f_mcore_correction = 1. / 3
                self.new_mcore = self.model.mcore + (self.mtot - self.mtot_calc) / const.mearth * f_mcore_correction # 07022019

                if self.verbosity > 1:
                    print ('current mcore (me) %f, total mass (g) = %g, missing mass (me) = %f, new core mass (me) = %f' % \
                        (self.model.mcore, self.mtot_calc, (self.mtot - self.mtot_calc) / const.mearth, self.new_mcore))

                if self.new_mcore <= 0.:
                    self.new_mcore = 2. / 3 * self.model.mcore
                elif self.new_mcore > 40.:
                    self.new_mcore = 0.5 * (self.mcore + 30.)

                if self.new_mcore <= 1e-2:
                    raise ongp.UnphysicalParameterError('model wants negative core mass')

                self.adjust_three_layer()
                
            elif self.params['model_type'] == 'linear_yz':
                reldiff = (self.mtot_calc - self.mtot) / self.mtot
                self.new_c = self.model.c - reldiff
                if self.new_c < 0.: raise ToFAdjustError(f'linear_yz wants c={self.new_c} < 0.')
                
                self.adjust_linear_yz()
                
            elif self.params['model_type'] == 'cosine_yz':
                
                self.model.mhe = trapz(self.model.y, x=self.m_calc)
                self.model.mz = trapz(self.model.z, x=self.m_calc)
                self.model.mean_y_xy = self.model.mhe / (self.mtot_calc - self.model.mz) # current actual M_He / (M_H + M_He)
                self.model.ym = self.model.mhe / self.mtot_calc # current actual M_He / M_tot
                                
                if not hasattr(self, 'y1_xy_prev'): self.y1_xy_prev = np.array([])
                if not hasattr(self, 'mean_y_xy_prev'): self.mean_y_xy_prev = np.array([])
                if not hasattr(self, 'c_prev'): self.c_prev = np.array([])
                if not hasattr(self, 'mtot_calc_prev'): self.mtot_calc_prev = np.array([])                
                self.y1_xy_prev = np.append(self.y1_xy_prev, self.model.y1_xy)
                self.mean_y_xy_prev = np.append(self.mean_y_xy_prev, self.model.mean_y_xy)
                self.c_prev = np.append(self.c_prev, self.model.c)
                self.mtot_calc_prev = np.append(self.mtot_calc_prev, self.mtot_calc)    
                    
                mtot_err = self.mtot - self.mtot_calc
                mtot_relerr = (self.mtot - self.mtot_calc) / self.mtot

                ymean_err = self.params['mean_y_xy'] - self.model.mean_y_xy
                ymean_relerr = (self.params['mean_y_xy'] - self.model.mean_y_xy) / self.params['mean_y_xy']
                
                if self.outer_iteration < 50:
                    self.model.c += mtot_relerr # what we usually do

                    # this is what we normally do and it works well if y1_xy is not close to zero
                    f = 1 if self.outer_iteration > 20 else 1.
                    self.model.y1_xy *= (1. + f * ymean_relerr)
                else: # try and get a better guess of the appropriate corrections for c and y1_xy simultaneously
                    
                    dmtot_dc = (np.diff(self.mtot_calc_prev) / np.diff(self.c_prev))[-1]
                    c_corr = mtot_err / dmtot_dc
                    self.model.c += c_corr
                    
                    dym_dy1 = (np.diff(self.mean_y_xy_prev) / np.diff(self.y1_xy_prev))[-1]
                    y1_corr = ymean_err / dym_dy1
                    self.model.y1_xy += y1_corr
                        
                # sanity checks to make sure values make sense, and give up if needed                
                if self.model.c < 0. and self.outer_iteration > 20: 
                    raise ToFAdjustError(f'cosine_yz wants c={self.model.c} < 0.')
                elif self.model.c > 1.:
                    raise ToFAdjustError(f'cosine_yz wants c={self.model.c} > 1. model might have an unrealistically low total z content.')
                elif self.model.c < 0.:
                    try:
                        self.model.c = 0.5 * self.model.c_prev[-1]
                    except AttributeError:
                        self.model.c = 0.1
                    
                if self.model.y1_xy < 0. and self.outer_iteration > 20:
                    raise ToFAdjustError(f'cosine_yz wants y1_xy={self.model.y1_xy} < 0.')
                elif self.model.y1_xy < 0.:
                    self.model.y1_xy = 0.5 * self.y1_xy_prev[-1]
                elif self.model.y1_xy < 1e-2 and self.outer_iteration > 20:
                    raise ToFAdjustError(f'cosine_yz wants y1_xy < 0.01')
                    
                self.adjust_cosine_yz()

                if 'debug_adjust' in self.params and self.params['debug_adjust']:
                    import pickle
                    debug = {'y_xy':self.model.y_xy, 'mean_y_xy':self.model.mean_y_xy, 'y1_xy':self.model.y1_xy, 'm':self.m_calc, 'c':self.model.c, 'mtot_calc':self.mtot_calc}
                    pickle.dump(debug, open(f'debug_{self.outer_iteration:02n}.pkl', 'wb'))
                
            else:
                raise ValueError('model_type {} not recognized'.format(self.params['model_type']))

            # reintegrate continuity for updated mass distribution
            dm = 4. / 3 * np.pi * self.rho[1:] * (self.l[1:] ** 3 - self.l[:-1] ** 3)
            dm = np.insert(dm, 0, 4. / 3 * np.pi * self.rho[0] * self.l[0] ** 3)
            self.m_calc = np.cumsum(dm)
            self.mtot_calc = self.m_calc[-1]

            # print('{:30} {:16.8e}'.format('outer loop after m_calc', np.dot(self.model.y, dm))) # checking mhe at various points

            # this is rhobar for all purposes except for the rhobar appearing in U(A0).
            self.rhobar = self.mtot_calc * 3. / 4 / np.pi / self.l[-1] ** 3

            if self.verbosity > 2:
                print( 'after adjust mcore: mtot_calc, reldiff', self.mtot_calc, abs(self.mtot_calc - self.mtot) / self.mtot)
                print( 'rhobar', self.rhobar)

            # bookkeeping and collecting things for real-time output
            self.et_total = time.time() - time_start_outer
            if self.params['model_type'] == 'three_layer':
                adjust_output = {
                    'mcore':self.model.mcore,
                    'y1':self.model.y1,
                    'y2':self.model.y2,
                    'mean_y_xy':self.model.mean_y_xy if hasattr(self.model, 'mean_y_xy') else -1,
                    }
            elif self.params['model_type'] == 'cosine_yz' or self.params['model_type'] == 'linear_yz':
                adjust_output = {
                    'c':self.model.c,
                    'y1_xy':self.model.y1_xy if hasattr(self.model, 'y1_xy') else -1,
                    'mean_y_xy':self.model.mean_y_xy if hasattr(self.model, 'mean_y_xy') else -1,
                    }
            elif self.params['model_type'] == 'cosine_yz_c':
                adjust_output = {
                    'z2':self.model.z2,
                    'y1_xy':self.model.y1_xy if hasattr(self.model, 'y1_xy') else -1,
                    'mean_y_xy':self.model.mean_y_xy if hasattr(self.model, 'mean_y_xy') else -1,
                    }
            elif 'dual_cavity' in self.params['model_type']:
                adjust_output = {
                    'r1':self.model.r1,
                    'r2':self.model.r2,
                    'r3':self.model.r3,
                    'y1':self.model.y1 if hasattr(self.model, 'y1') else -1,
                    'y1_xy':self.model.y1_xy if hasattr(self.model, 'y1_xy') else -1,
                    'mean_y_xy':self.model.mean_y_xy if hasattr(self.model, 'mean_y_xy') else -1,
                    }
            else:
                # etc etc
                pass
            if self.verbosity > 1 or (self.outer_iteration == 0 and self.verbosity > 0):
                names = 'iout', 'iin', 'rhobar', 'mtot_calc', \
                    *list(adjust_output), \
                    'z2', 'z1', 'nz', \
                    'et_total', 'j2', 'j4', 'j6', 'req', 'rm', 'rpol', 'small'
                print (('%5s ' * 2 + '%11s ' * (len(names) - 2)) % names)
            if self.verbosity > 0:
                data = self.outer_iteration, self.inner_iteration, self.rhobar, \
                        self.mtot_calc, \
                        *[adjust_output[key] for key in list(adjust_output)], \
                        self.model.z2 if hasattr(self.model, 'z2') else -1, \
                        self.model.z1, self.nz, \
                        self.et_total, self.j2, self.j4, self.j6, self.r_eq[-1], self.rm, self.r_pol[-1], self.small,
                print (('%5i ' * 2 + '%11.5g ' * (len(data) - 2)) % data)
                if 'debug_outer' in list(self.params) and self.params['debug_outer']:
                    import pickle
                    with open('debug_{:03n}.pkl'.format(self.outer_iteration), 'wb') as fout:
                        pickle.dump({'l':self.l, 'p':self.model.p, 't':self.model.t, 'y':self.model.y, 'm_calc':self.m_calc, 'j2':self.j2, 'j4':self.j4, 'j6':self.j6, 'req':self.r_eq[-1]}, fout)

            # scale mean radii to match surface equatorial radius
            # if self.outer_iteration < 20:
            self.l *= (self.params['req'] / self.r_eq[-1])
            # self.l *= (58232e5 / self.l[-1]) # experimenting 06062019
            # factor = self.params['req'] / self.r_eq[-1]
            # self.l *= (0.9 + 0.1 * factor)
            self.rm_old = self.rm
            self.rm = self.l[-1]

            if 'adjust_small' in list(self.params) and self.params['adjust_small']:
                # originally small assumed saturn's true mtot and rm; scale small to reflect that
                # original rotation rate but at the present model mtot and rm.
                self.small = self.small_start * (self.rm / const.rsat) ** 3
                # pass
                # m = (omega_rot / omega_dyn) ** 2 = omega_rot**2 * R ** 3 / GM
                # omega_rot ** 2 = constant = m * gm / r ** 3 = mout * gmout / rout ** 3
                # mout = m * (gm / gmout) * (rout / r) ** 3
                # self.small = self.small_start / (self.rm / const.rsat) ** 3 / (const.msat / self.mtot_calc) # mass error insignificant

            if self.outer_iteration > 10:
                # if change in all j2n since last outer iteration is within tolerance,
                # there's nothing to be gained from adjusting the mass distribution--we're done.
                if np.all(abs((self.j2n_last_outer - self.j2n) / (self.j2n+1e-20)) < self.j2n_rtol) \
                    and self.outer_iteration > 10 \
                    and self.inner_done \
                    and abs(self.mtot_calc - self.mtot) / self.mtot < self.mtot_rtol \
                    and abs(self.model.mean_y_xy - self.params['mean_y_xy']) / self.params['mean_y_xy'] < self.ymean_rtol:

                    if self.verbosity >= 3.:
                        print ('terminate outer loop; all dj2n/j2n < %g.' % self.j2n_rtol)

                    self.outer_done = True

                    self.set_s2n(force_full=True)

                    self.save_model_summary()

                    break

            self.j2n_last_outer = self.j2n

        return

    def adjust_three_layer(self):
        '''
        enforce new mcore, set y and z, reintegrate *adiabatic* t profile, and recalculate densities.
        this will not in general conserve the total mass of the model; the outer loop itself handles this by
        comparing mtot to mtot_calc and telling this routine what mcore to set.
        '''

        old_mcore = self.model.old_mcore = self.model.mcore
        self.model.old_kcore = self.model.kcore
        mcore = self.model.mcore = self.new_mcore
        kcore = self.model.kcore = np.where(self.m_calc < mcore * const.mearth)[0][-1]

        self.model.dm = np.diff(self.m_calc, prepend=self.m_calc[0])
        if (self.outer_iteration > 10 and abs(kcore - self.model.old_kcore) <= 5) or self.outer_iteration > 50:
            # later in iterations, add a zone to help pinpoint the correct core mass
            alpha_m = (mcore * const.mearth - self.m_calc[kcore]) / (self.m_calc[kcore+1] - self.m_calc[kcore])
            assert alpha_m > 0.
            self.m_calc = np.insert(self.m_calc, kcore, mcore * const.mearth)
            self.p = np.insert(self.p, kcore+1, alpha_m * self.p[kcore + 1] + (1. - alpha_m) * self.p[kcore])
            self.l = np.insert(self.l, kcore+1, alpha_m * self.l[kcore + 1] + (1. - alpha_m) * self.l[kcore])
            self.rho = np.insert(self.rho, kcore, 0.) # actual value calculated below

            self.model.dm = np.diff(self.m_calc, prepend=self.m_calc[0])
            self.model.grada = np.insert(self.model.grada, kcore, self.model.grada[kcore])
            self.model.gradt = np.insert(self.model.gradt, kcore, self.model.gradt[kcore])
            self.model.t = np.insert(self.model.t, kcore, self.model.t[kcore])
            self.model.y = np.insert(self.model.y, kcore, self.model.y[kcore])
            self.model.z = np.insert(self.model.z, kcore, self.model.z[kcore])

            self.add_point_to_figure_functions(kcore)
            self.nz += 1
            kcore += 1
            self.model.kcore = kcore

        if np.any(np.diff(self.l) <= 0.):
            raise ValueError('radius not monotone increasing')

        ktrans = self.model.ktrans = np.where(self.p > self.model.ptrans * 1e12)[0][-1] # updated p

        t = np.zeros_like(self.p)
        y = np.zeros_like(self.p)
        z = np.zeros_like(self.p)

        z[:kcore] = 1.
        # z[kcore:ktrans] = self.model.z2
        # z[ktrans:] = self.model.z1
        z[kcore:] = self.model.z1
        if hasattr(self.model, 'z2'):
            z[kcore:ktrans] = self.model.z2
        self.model.z = z

        self.model.y[:kcore] = 0.
        if hasattr(self.model, 'y2'):
            self.model.y[kcore:ktrans] = self.model.y2
            self.model.y[ktrans:] = self.model.y1
        else: # first outer iteration
            self.model.y[kcore:] = self.model.y1
        
        y = self.model.y
        assert not np.any(y[kcore:] == 0)
        # self.model.k_shell_top now set; compare to ptrans

        t[-1] = self.model.t1

        try:
            assert not np.any(np.isnan(self.p)), 'nans in p during adjust mcore, before t integration'
            assert not np.any(np.isnan(t)), 'nans in t during adjust mcore, before t integration'
            assert not np.any(np.isnan(self.model.grada)), 'nans in grada during adjust mcore, before t integration'
        except AssertionError as e:
            raise ongp.EOSError(e.args[0])

        self.model.gradt[kcore:] = self.model.grada[kcore:] \
            = self.model.hhe_eos.get_grada(np.log10(self.p[kcore:]), np.log10(self.model.t[kcore:]), y[kcore:])
        self.model.gradt[:kcore] = 0.
        self.model.grada[:kcore] = 0.

        if np.any(np.isnan(self.model.grada)):
            raise ValueError('nans in grada: y1={} y2={} z1={} z2={}'.format(self.model.y1, self.model.y2, self.model.z1, self.model.z2))

        t = np.exp(cumtrapz(self.model.gradt[::-1] / self.p[::-1], x=self.p[::-1], initial=0.))[::-1]
        t *= self.model.t1

        self.rho[:kcore] = self.model.get_rho_z(np.log10(self.p[:kcore]), np.log10(t[:kcore]))
        self.rho[kcore:] = self.model.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(t[kcore:]), y[kcore:], z[kcore:])

        self.model.t = t
        self.model.y = y
        self.model.z = z

        self.model.mhe = np.dot(self.model.dm, self.model.y)
        self.model.mz = np.dot(self.model.dm, self.model.z)
        self.model.mean_y_xy = self.model.mhe / (self.mtot_calc - self.model.mz) # current actual M_He / (M_H + M_He)
        self.model.ym = self.model.mhe / self.mtot_calc # current actual M_He / M_tot

        if 'ym' in list(self.params):
            assert 'mean_y_xy' not in list(self.params), 'must specify only one of ym and mean_y_xy in tof_params.'
            relative_mismatch = (self.params['ym'] - self.model.ym) / self.params['ym']
        elif 'mean_y_xy' in list(self.params):
            relative_mismatch = (self.params['mean_y_xy'] - self.model.mean_y_xy) / self.params['mean_y_xy']
        else:
            raise ValueError('no rule to adjust helium mass fraction; must set either ym or mean_y_xy.')
        # adjust y2 to approach correct ymean
        if not hasattr(self.model, 'y2'): self.model.y2 = self.model.y1
        self.model.y2 *= (1. + relative_mismatch)

        return

    def adjust_linear_yz(self):
        
        '''
        similar but for a model with continuous Y and Z distributions. 
        Z and Y_XY each increase linearly from r/R = c+w/2 down to c-w/2.
        '''

        old_c = self.model.old_c = self.model.c
        c = self.model.c = self.new_c

        kcore = 0

        if np.any(np.diff(self.l) <= 0.):
            raise ValueError('radius not monotone increasing')

        w = self.model.w

        # this time y gradient (y1_xy to y2_xy) coincides exactly with z gradient (z1 to z2)
        rf = self.l / self.l[-1]

        self.model.z = self.model.z1 + (self.model.z2 - self.model.z1) * (0.5 + (c - rf) / w)
        self.model.z[rf < c - w / 2] = self.model.z2
        self.model.z[rf > c + w / 2] = self.model.z1
        z = self.model.z

        if self.model.y2_xy is None:
            y_xy = self.model.y1_xy * np.ones_like(rf)
        else:
            y_xy = self.model.y1_xy + (self.model.y2_xy - self.model.y1_xy) * (0.5 + (c - rf) / w)
            y_xy[rf < c - w / 2] = self.model.y2_xy
            y_xy[rf > c + w / 2] = self.model.y1_xy
        self.model.y = y = y_xy * (1. - z)
        self.model.y[self.model.y == 0.] = 1e-20 # keeps mixture eos evaluable

        assert np.all(self.model.y >= 0)
        assert np.all(self.model.y < 1)

        if np.any(np.isnan(self.model.grada)):
            print('nans in grada: y1={} y2={} z1={} z2={}'.format(self.model.y1, self.model.y2, self.model.z1, self.model.z2))
            raise ValueError

        try:
            assert not np.any(np.isnan(self.p)), 'nans in p during adjust mcore, before t integration'
            assert not np.any(np.isnan(self.model.t)), 'nans in t during adjust mcore, before t integration'
            assert not np.any(np.isnan(self.model.grada)), 'nans in grada during adjust mcore, before t integration'
        except AssertionError as e:
            raise ongp.EOSError(e.args[0])

        if 'rrho_in_composition_gradient' in self.params and self.params['rrho_in_composition_gradient']:
            # get grady, gradz, dlnrho/dlny, dlnrho/dlnz, set gradt > grada
            hhe_eos = self.model.hhe_eos
            z_eos = self.model.z_eos
            rho = self.rho

            # call eos for new quantities not saved from tof
            hhe_res = hhe_eos.get(np.log10(self.p), np.log10(self.model.t), y)
            # assert self.model.z_eos_option == 'reos water'
            if self.model.z_eos_option == 'reos water':
                import reos_water_rhot
                z_eos_rhot = reos_water_rhot.eos()
            elif self.model.z_eos_option.split()[0] == 'aneos':
                import aneos_rhot
                z_eos_rhot = aneos_rhot.eos(self.model.z_eos_option.split()[1])
            else:
                raise ValueError('good z eos derivs only implemented for reos water or aneos.')
            z_res = z_eos_rhot.get(np.log10(rho), np.log10(self.model.t))

            rho_hhe = 10 ** hhe_res['logrho']
            rho_h = hhe_res['rho_h']
            rho_he = hhe_res['rho_he']
            assert not np.any(z <= 0)
            rho_z = z * rho * rho_hhe / (rho_hhe - rho * (1. - z))
            chi_z = rho * z * (1. / rho_hhe - 1. / rho_z) # chi_z is a bad name, this is actually dlnrho_dlnz_const_pty
            chi_y = rho * y * (1. - z) * (1. / rho_h - 1. / rho_he) # dlnrho_dlny_const_ptz

            chi_rho_hhe = hhe_res['chirho']
            chi_rho_z = z_res['chirho']

            if self.model.z_eos_option.split()[0] == 'aneos':
                # ignore the z part of chi_t and chi_rho
                chi_rho = chi_rho_hhe
                chi_t = - hhe_res['rhot'] / hhe_res['rhop']
                dlnrho_dlnt_const_p = - hhe_res['rhot']
            else:
                if False:
                    chi_rho = (z * rho / rho_z / chi_rho_z + (1. - z) * rho / rho_hhe / chi_rho_hhe) ** -1.

                    dlnrho_dlnt_const_p = z * rho / rho_z * z_res['rhot'] + (1. - z) * rho / rho_he * hhe_res['rhot']
                    dlnrho_dlnp_const_t = z * rho / rho_z * z_res['rhop'] + (1. - z) * rho / rho_he * hhe_res['rhop']
                    chi_t = - dlnrho_dlnt_const_p / dlnrho_dlnp_const_t
                else:
                    chi_rho = chi_rho_hhe
                    chi_t = - hhe_res['rhot'] / hhe_res['rhop']

            grady = np.diff(np.log(y), append=0) / np.diff(np.log(self.p), append=0)
            gradz = np.diff(np.log(z), append=0) / np.diff(np.log(self.p), append=0)
            self.model.brunt_b = b = chi_rho / chi_t * (chi_y * grady + chi_z * gradz)
            self.model.grada = hhe_res['grada']
            self.model.gradt = self.model.grada + self.model.brunt_b * self.params['rrho_in_composition_gradient']

            t = np.exp(cumtrapz(self.model.gradt[::-1] / self.p[::-1], x=self.p[::-1], initial=0.))[::-1]
            t *= self.model.t1
            self.model.t = t
        else: # simply assume gradt = grada and integrate
            self.model.gradt = self.model.grada = self.model.hhe_eos.get_grada(np.log10(self.p), np.log10(self.model.t), y)
            t = np.exp(cumtrapz(self.model.gradt[::-1] / self.p[::-1], x=self.p[::-1], initial=0.))[::-1]
            t *= self.model.t1
            self.model.t = t

        self.rho = self.model.get_rho_xyz(np.log10(self.p), np.log10(self.model.t), self.model.y, self.model.z)

        if np.any(np.isnan(self.rho)):
            raise ToFAdjustError('nans in rho during adjust; possibly off eos tables')
            print('logp where rho==nan', np.log10(self.p[np.isnan(self.rho)]))
            print('logt where rho==nan', np.log10(self.model.t[np.isnan(self.rho)]))
            print('chi_rho where rho==nan', chi_rho[np.isnan(self.rho)])
            print('chi_t where rho==nan', chi_t[np.isnan(self.rho)])
            print('chi_y where rho==nan', chi_y[np.isnan(self.rho)])
            print('chi_z where rho==nan', chi_z[np.isnan(self.rho)])
            print('grady where rho==nan', grady[np.isnan(self.rho)])
            print('gradz where rho==nan', gradz[np.isnan(self.rho)])
            print('z where rho==nan', self.model.z[np.isnan(self.rho)])
            print('y where rho==nan', self.model.y[np.isnan(self.rho)])
            print('z where rho==nan', self.model.z[np.isnan(self.rho)])

        self.model.mhe = np.dot(self.dm, self.model.y)
        self.model.mz = np.dot(self.dm, self.model.z)
        self.model.mean_y_xy = self.model.mhe / (self.mtot_calc - self.model.mz) # current actual M_He / (M_H + M_He)
        self.model.ym = self.model.mhe / self.mtot_calc # current actual M_He / M_tot

        if 'ym' in list(self.params):
            relative_mismatch = (self.params['ym'] - self.model.ym) / self.params['ym']
        elif 'mean_y_xy' in list(self.params):
            relative_mismatch = (self.params['mean_y_xy'] - self.model.mean_y_xy) / self.params['mean_y_xy']

        f = 0.5 if self.outer_iteration > 20 else 1.
        # self.model.dboty *= (1. + f * relative_mismatch)
        # adjust outer envelope helium abundance to approach plausible mean abundance
        self.model.y1_xy *= (1. + f * relative_mismatch)

        # print(self.model.mean_y_xy, self.params['mean_y_xy'], relative_mismatch, self.model.dboty)

        return

    def adjust_cosine_yz(self):
        '''
        similar but for a model with continuous Y and Z distributions. 
        Z and Y_XY each increase sinusoidally from r/R = c+w/2 down to c-w/2.
        '''

        if np.any(np.diff(self.l) <= 0.):
            raise ValueError('radius not monotone increasing')

        c = self.model.c
        w = self.model.w

        # this time y gradient (y1_xy to y2_xy) coincides exactly with z gradient (z1 to z2)
        rf = self.l / self.l[-1]

        self.model.z = self.model.z1 + (self.model.z2 - self.model.z1) * 0.5 * (1. + np.cos(np.pi * (c - rf - w/2) / w))
        self.model.z[rf < c - w / 2] = self.model.z2
        self.model.z[rf > c + w / 2] = self.model.z1
        z = self.model.z

        if self.model.y2_xy is None:
            y_xy = self.model.y1_xy * np.ones_like(rf)
        else:
            y_xy = self.model.y1_xy + (self.model.y2_xy - self.model.y1_xy) * 0.5 * (1. + np.cos(np.pi * (c - rf - w/2) / w))
            y_xy[rf < c - w / 2] = self.model.y2_xy
            y_xy[rf > c + w / 2] = self.model.y1_xy
        self.model.y = y = y_xy * (1. - z)
        self.model.y_xy = y_xy
        self.model.y[self.model.y == 0.] = 1e-20 # keeps mixture eos evaluable

        assert np.all(self.model.y >= 0)
        assert np.all(self.model.y < 1)

        if np.any(np.isnan(self.model.grada)):
            print('nans in grada: y1={} y2={} z1={} z2={}'.format(self.model.y1, self.model.y2, self.model.z1, self.model.z2))
            raise ValueError

        try:
            assert not np.any(np.isnan(self.p)), 'nans in p during adjust mcore, before t integration'
            assert not np.any(np.isnan(self.model.t)), 'nans in t during adjust mcore, before t integration'
            assert not np.any(np.isnan(self.model.grada)), 'nans in grada during adjust mcore, before t integration'
        except AssertionError as e:
            raise ongp.EOSError(e.args[0])

        if 'rrho_in_composition_gradient' in self.params and self.params['rrho_in_composition_gradient']:
            # get grady, gradz, dlnrho/dlny, dlnrho/dlnz, set gradt > grada
            hhe_eos = self.model.hhe_eos
            z_eos = self.model.z_eos
            rho = self.rho

            # new quantities not saved from tof
            hhe_res = hhe_eos.get(np.log10(self.p), np.log10(self.model.t), y)
            # assert self.model.z_eos_option == 'reos water'
            if self.model.z_eos_option == 'reos water':
                import reos_water_rhot
                z_eos_rhot = reos_water_rhot.eos()
            elif self.model.z_eos_option.split()[0] == 'aneos':
                import aneos_rhot
                z_eos_rhot = aneos_rhot.eos(self.model.z_eos_option.split()[1])
            else:
                raise ValueError('good z eos derivs only implemented for reos water or aneos.')
            z_res = z_eos_rhot.get(np.log10(rho), np.log10(self.model.t))

            rho_hhe = 10 ** hhe_res['logrho']
            rho_h = hhe_res['rho_h']
            rho_he = hhe_res['rho_he']
            assert not np.any(z <= 0)
            rho_z = z * rho * rho_hhe / (rho_hhe - rho * (1. - z))
            chi_z = rho * z * (1. / rho_hhe - 1. / rho_z) # chi_z is a bad name, this is actually dlnrho_dlnz_const_pty
            chi_y = rho * y * (1. - z) * (1. / rho_h - 1. / rho_he) # dlnrho_dlny_const_ptz

            chi_rho_hhe = hhe_res['chirho']
            chi_rho_z = z_res['chirho']

            if self.model.z_eos_option.split()[0] == 'aneos':
                # ignore the z part of chi_t and chi_rho
                chi_rho = chi_rho_hhe
                chi_t = - hhe_res['rhot'] / hhe_res['rhop']
                dlnrho_dlnt_const_p = - hhe_res['rhot']
            else:
                if False:
                    chi_rho = (z * rho / rho_z / chi_rho_z + (1. - z) * rho / rho_hhe / chi_rho_hhe) ** -1.

                    dlnrho_dlnt_const_p = z * rho / rho_z * z_res['rhot'] + (1. - z) * rho / rho_he * hhe_res['rhot']
                    dlnrho_dlnp_const_t = z * rho / rho_z * z_res['rhop'] + (1. - z) * rho / rho_he * hhe_res['rhop']
                    chi_t = - dlnrho_dlnt_const_p / dlnrho_dlnp_const_t
                else:
                    chi_rho = chi_rho_hhe
                    chi_t = - hhe_res['rhot'] / hhe_res['rhop']

            grady = np.diff(np.log(y), append=0) / np.diff(np.log(self.p), append=0)
            gradz = np.diff(np.log(z), append=0) / np.diff(np.log(self.p), append=0)
            self.model.brunt_b = b = chi_rho / chi_t * (chi_y * grady + chi_z * gradz)
            self.model.grada = hhe_res['grada']
            self.model.gradt = self.model.grada + self.model.brunt_b * self.params['rrho_in_composition_gradient']

            t = np.exp(cumtrapz(self.model.gradt[::-1] / self.p[::-1], x=self.p[::-1], initial=0.))[::-1]
            t *= self.model.t1
            self.model.t = t
        else:
            try:
                self.model.gradt = self.model.grada = self.model.hhe_eos.get_grada(np.log10(self.p), np.log10(self.model.t), y)
            except:
                raise ongp.EOSError('failed in eos call in adjust_cosine_yz')
            t = np.exp(cumtrapz(self.model.gradt[::-1] / self.p[::-1], x=self.p[::-1], initial=0.))[::-1]
            t *= self.model.t1
            self.model.t = t

        self.rho = self.model.get_rho_xyz(np.log10(self.p), np.log10(self.model.t), self.model.y, self.model.z)

        if np.any(np.isnan(self.rho)):
            raise ToFAdjustError('nans in rho during adjust; possibly off eos tables')
            print('logp where rho==nan', np.log10(self.p[np.isnan(self.rho)]))
            print('logt where rho==nan', np.log10(self.model.t[np.isnan(self.rho)]))
            print('chi_rho where rho==nan', chi_rho[np.isnan(self.rho)])
            print('chi_t where rho==nan', chi_t[np.isnan(self.rho)])
            print('chi_y where rho==nan', chi_y[np.isnan(self.rho)])
            print('chi_z where rho==nan', chi_z[np.isnan(self.rho)])
            print('grady where rho==nan', grady[np.isnan(self.rho)])
            print('gradz where rho==nan', gradz[np.isnan(self.rho)])
            print('z where rho==nan', self.model.z[np.isnan(self.rho)])
            print('y where rho==nan', self.model.y[np.isnan(self.rho)])
            print('z where rho==nan', self.model.z[np.isnan(self.rho)])
        
        return

    def add_point_to_figure_functions(self, k):
        self.s0 = np.insert(self.s0, k, self.s0[k])
        self.s2 = np.insert(self.s2, k, self.s2[k])
        self.s4 = np.insert(self.s4, k, self.s4[k])
        self.s6 = np.insert(self.s6, k, self.s6[k])
        self.s8 = np.insert(self.s8, k, self.s8[k])

        self.f0 = np.insert(self.f0, k, self.f0[k])
        self.f2 = np.insert(self.f2, k, self.f2[k])
        self.f4 = np.insert(self.f4, k, self.f4[k])
        self.f6 = np.insert(self.f6, k, self.f6[k])
        self.f8 = np.insert(self.f8, k, self.f8[k])

        self.f0p = np.insert(self.f0p, k, self.f0p[k])
        self.f2p = np.insert(self.f2p, k, self.f2p[k])
        self.f4p = np.insert(self.f4p, k, self.f4p[k])
        self.f6p = np.insert(self.f6p, k, self.f6p[k])
        self.f8p = np.insert(self.f8p, k, self.f8p[k])

        self.ss0 = np.insert(self.ss0, k, self.ss0[k])
        self.ss2 = np.insert(self.ss2, k, self.ss2[k])
        self.ss4 = np.insert(self.ss4, k, self.ss4[k])
        self.ss6 = np.insert(self.ss6, k, self.ss6[k])
        self.ss8 = np.insert(self.ss8, k, self.ss8[k])

        self.ss0p = np.insert(self.ss0p, k, self.ss0p[k])
        self.ss2p = np.insert(self.ss2p, k, self.ss2p[k])
        self.ss4p = np.insert(self.ss4p, k, self.ss4p[k])
        self.ss6p = np.insert(self.ss6p, k, self.ss6p[k])
        self.ss8p = np.insert(self.ss8p, k, self.ss8p[k])

    def save_model_summary(self):
        """saves tof parameter/scalar/vector data."""
        import pickle

        # write a record of warnings if this model is not up to snuff for one reason or another
        relative_mass_error = abs((self.mtot_calc - self.mtot)/self.mtot)
        if relative_mass_error > self.mtot_rtol or not self.inner_done or not self.outer_done:
            if relative_mass_error > self.mtot_rtol:
                raise ConvergenceError('attempted to save model summary for model with mass error exceeding specified tolerance. relerr, rtol: %g %g' % (relative_mass_error, self.mtot_rtol))
            if not self.inner_done:
                raise ConvergenceError('attempted to save model summary with inner_done == False')
            if not self.outer_done:
                raise ConvergenceError('attempted to save model summary with outer_done == False')

        scalars = {}
        scalars['uid'] = self.uid # unique identifier
        scalars['nz'] = self.nz
        scalars['j2'] = self.j2
        scalars['j4'] = self.j4
        scalars['j6'] = self.j6
        scalars['j8'] = self.j8
        scalars['req'] = self.r_eq[-1]
        scalars['rpol'] = self.r_pol[-1]
        scalars['rm'] = self.rm
        scalars['rhobar'] = self.rhobar
        scalars['mtot_calc'] = self.mtot_calc
        scalars['small'] = self.small
        scalars['q'] = self.q
        scalars['et_total'] = self.et_total
        scalars['omega0'] = np.sqrt(const.cgrav * self.mtot_calc / self.rm ** 3)

        self.model.mhe = trapz(self.model.y, x=self.m_calc) # np.dot(self.dm, self.model.y)
        self.model.mz = trapz(self.model.z, x=self.m_calc) # np.dot(self.dm, self.model.z)
        self.model.mean_y_xy = self.model.mhe / (self.mtot_calc - self.model.mz) # current actual M_He / (M_H + M_He)
        self.model.ym = self.model.mhe / self.mtot_calc # current actual M_He / M_tot

        # esimates of core mass and core radius, written with a smooth model in mind;
        # in the case of a three-layer model should reduce to the specified mcore
        scalars['mcore_proxy'] = self.m_calc[self.model.z < 0.5][0]
        scalars['rcore_proxy'] = self.l[self.model.z < 0.5][0]
        mz = cumtrapz(self.model.z, x=self.m_calc, initial=0.)
        try:
            scalars['mz_in'] = mz[self.model.z > 0.5][-1]
        except IndexError:
            raise ToFAdjustError('no zone with Z>0.5')
        scalars['mz_out'] = mz[-1] - scalars['mz_in']
        my = cumtrapz(self.model.y, x=self.m_calc, initial=0.)
        scalars['my_in'] = my[self.model.z > 0.5][-1]
        scalars['my_out'] = my[-1] - scalars['my_in']
        scalars['rhoc_proxy'] = self.rho[self.model.z < 0.5][0]
        scalars['pc_proxy'] = self.p[self.model.z < 0.5][0]
        scalars['tc_proxy'] = self.model.t[self.model.z < 0.5][0]

        # scalar quantities that self.model may or may not have set, depending on model type:
        model_scalar_names = 'mcore', 'y1', 'y2', 'ym', 'kcore', 'k_shell_top', 'ktrans', \
            'z1', 'z2', 'mhe', 'mz', 'c', 'c2', 'y1_xy', 'y2_xy', 'max_y', 'r1', 'r2', 'r3'
        for name in model_scalar_names:
            if hasattr(self.model, name): scalars[name] = getattr(self.model, name)

        self.scalars = scalars # save as attribute so that MCMC can use for likelihood

        output = {}
        for attr in 'params', 'mesh_params':
            output[attr] = getattr(self, attr)
        output['scalars'] = scalars

        # gather vector output
        vectors = {}
        vectors['l'] = l = self.l
        vectors['req'] = self.r_eq
        vectors['rpol'] = self.r_pol
        vectors['rho'] = rho = self.rho
        vectors['p'] = p = self.p
        # vectors['u'] = self.u # rarely care
        vectors['m_calc'] = m = self.m_calc
        # was skipping these shape functions for a while # am skipping them again to save disk space
        # vectors['s0'] = self.s0
        # vectors['s2'] = self.s2
        # vectors['s4'] = self.s4
        # vectors['s6'] = self.s6
        # vectors['s8'] = self.s8
        #
        # vectors['ss0'] = self.ss0
        # vectors['ss2'] = self.ss2
        # vectors['ss4'] = self.ss4
        # vectors['ss6'] = self.ss6
        # vectors['ss8'] = self.ss8

        # add model's y/z profiles; not a quantity tof itself is aware of, but of interest
        vectors['y'] = y = self.model.y
        vectors['z'] = z = self.model.z
        # and temperature
        vectors['t'] = t = self.model.t
        vectors['gradt'] = gradt = self.model.gradt
        y[y == 0.] = 1e-10

        # both forms of brunt calculation will require eos quantities
        hhe_eos = self.model.hhe_eos
        z_eos = self.model.z_eos
        # model = output # easily port following code from analyze_chains.ipynb

        # new quantities not saved from tof
        hhe_res = hhe_eos.get(np.log10(p), np.log10(t), y)
        # assert self.model.z_eos_option == 'reos water'
        if self.model.z_eos_option == 'reos water':
            import reos_water_rhot
            z_eos_rhot = reos_water_rhot.eos()
            z_res = z_eos_rhot.get(np.log10(rho), np.log10(t))
        elif self.model.z_eos_option.split()[0] == 'aneos':
            if self.model.z_eos_option.split()[1] == 'mix':
                pass # have implemented rhop, rhot, chirho, chit, but not gamma1/grada yet
            else:
                import aneos_rhot
                z_eos_rhot = aneos_rhot.eos(self.model.z_eos_option.split()[1])
                z_res = z_eos_rhot.get(np.log10(rho), np.log10(t))
        else:
            raise ValueError('good z eos derivs only implemented for reos water or aneos.')

        vectors['gamma1'] = hhe_res['gamma1']
        grada = vectors['grada'] = self.model.grada # hhe_res['grada']
        vectors['rhot'] = hhe_res['rhot']
        vectors['g'] = g = m / l ** 2 * const.cgrav

        vectors['dlnp_dr'] = np.diff(np.log(p), append=0) / np.diff(l, append=0)
        vectors['dlnrho_dr'] = np.diff(np.log(rho), append=0) / np.diff(l, append=0)

        vectors['n2_direct'] = g * (vectors['dlnp_dr'] / vectors['gamma1'] - vectors['dlnrho_dr'])

        rho_hhe = 10 ** hhe_res['logrho']
        rho_h = hhe_res['rho_h']
        rho_he = hhe_res['rho_he']
        rho_z = z * rho * rho_hhe / (rho_hhe - rho * (1. - z))
        chi_z = rho * z * (1. / rho_hhe - 1. / rho_z) # chi_z is a bad name, this is actually dlnrho_dlnz_const_pty
        chi_y = rho * y * (1. - z) * (1. / rho_h - 1. / rho_he) # dlnrho_dlny_const_ptz

        chi_rho_hhe = hhe_res['chirho']

        if self.model.z_eos_option.split()[0] == 'aneos':
            # ignore the z part of chi_t and chi_rho
            chi_rho = chi_rho_hhe
            chi_t = - hhe_res['rhot'] / hhe_res['rhop']
            dlnrho_dlnt_const_p = - hhe_res['rhot']
        else:
            chi_rho_z = z_res['chirho']
            chi_rho = (z * rho / rho_z / chi_rho_z + (1. - z) * rho / rho_hhe / chi_rho_hhe) ** -1.

            dlnrho_dlnt_const_p = z * rho / rho_z * z_res['rhot'] + (1. - z) * rho / rho_he * hhe_res['rhot']
            dlnrho_dlnp_const_t = z * rho / rho_z * z_res['rhop'] + (1. - z) * rho / rho_he * hhe_res['rhop']
            chi_t = - dlnrho_dlnt_const_p / dlnrho_dlnp_const_t

        if self.params['model_type'] is not 'three_layer':
            grady = np.diff(np.log(y), append=0) / np.diff(np.log(p), append=0)
            gradz = np.diff(np.log(z), append=0) / np.diff(np.log(p), append=0)
            vectors['brunt_b'] = b = chi_rho / chi_t * (chi_y * grady + chi_z * gradz)

            # more quantities for diagnostics; need not save in the long run
            vectors['grady'] = grady
            vectors['gradz'] = gradz
            vectors['chi_y'] = chi_y
            vectors['chi_z'] = chi_z
            vectors['chi_t'] = chi_t
            vectors['chi_rho'] = chi_rho
            vectors['delta'] = - dlnrho_dlnt_const_p # do save this one if going to save gyre info

            vectors['n2'] = g ** 2 * rho / p * chi_t / chi_rho * (grada - gradt + b)

            n2 = np.copy(vectors['n2']) # copy because it may be abused below (e.g., zero out n2<0)
            n2[n2 < 0] = 0.
            w0 = np.sqrt(const.cgrav * scalars['mtot_calc'] / scalars['rm'] ** 3)
            scalars['max_n'] = np.sqrt(max(n2)) / w0

            scalars['pi0'] = np.pi ** 2 * 2 / trapz(np.sqrt(n2), x=np.log(l))
            scalars['mean_n2_gmode_cavity'] = trapz(l * n2) / trapz(l)
        else: # special handling for all the zones with y or z or both equal to zero
            y[y == 0] = 1e-50
            z[z == 0] = 1e-50
            grady = np.diff(np.log(y), append=0) / np.diff(np.log(p), append=0)
            gradz = np.diff(np.log(y), append=0) / np.diff(np.log(p), append=0)
            vectors['brunt_b'] = b = chi_rho / chi_t * (chi_y * grady + chi_z * gradz)
            vectors['n2'] = g ** 2 * rho / p * chi_t / chi_rho * (grada - gradt + b)

            # more quantities for diagnostics; need not save in the long run
            # vectors['grady'] = grady
            # vectors['chi_y'] = chi_y
            # vectors['chi_z'] = chi_z
            vectors['chi_t'] = chi_t
            vectors['chi_rho'] = chi_rho
            vectors['delta'] = - dlnrho_dlnt_const_p # do save this one if going to save gyre info

        scalars['mean_y_xy'] = self.model.mean_y_xy

        output['vectors'] = self.vectors = vectors
        output['scalars'] = self.scalars = scalars
        outfile = '{}/tof4_data.pkl'.format(self.output_path)
        if os.path.exists(outfile):
            raise ValueError('{} unexpectedly already exists'.format(outfile))
        with open(outfile, 'wb') as f:
            pickle.dump(output, f)
        if self.verbosity > 0:
            print('wrote {}'.format(outfile))

    def set_req_rpol(self):
        '''
        calculate equatorial and polar radius vectors from the figure functions s_2n and legendre polynomials P_2n.
        see N17 eq. (B.1) or ZT78 eq. (27.1).
        also updates q from m and new r_eq[-1].
        '''

        # equator: mu = cos(pi/2) = 0
        self.r_eq = self.l * (1. + self.s0 * self.pp0(0.) \
                                 + self.s2 * self.pp2(0.) \
                                 + self.s4 * self.pp4(0.) \
                                 + self.s6 * self.pp6(0.) \
                                 + self.s8 * self.pp8(0.))

        # pole: mu = cos(0) = 1
        self.r_pol = self.l * (1. + self.s0 * self.pp0(1.) \
                                  + self.s2 * self.pp2(1.) \
                                  + self.s4 * self.pp4(1.) \
                                  + self.s6 * self.pp6(1.) \
                                  + self.s8 * self.pp8(1.))

        self.q = self.small * (self.r_eq[-1] / self.rm) ** 3


    def set_ss2n_ss2np(self):
        '''
        N17 eq. (B.9).
        '''

        self.z = self.l / self.rm

        ss2_integral = cumtrapz(self.z ** (2. + 3) * self.f2 / self.rhobar, x=self.rho, initial=0.)
        ss4_integral = cumtrapz(self.z ** (4. + 3) * self.f4 / self.rhobar, x=self.rho, initial=0.)
        ss6_integral = cumtrapz(self.z ** (6. + 3) * self.f6 / self.rhobar, x=self.rho, initial=0.)
        ss8_integral = cumtrapz(self.z ** (8. + 3) * self.f8 / self.rhobar, x=self.rho, initial=0.)

        # integrals from 0 to z
        ss0p_integral = cumtrapz(self.z ** (2. - 0) * self.f0p / self.rhobar, x=self.rho, initial=0.)
        ss2p_integral = cumtrapz(self.z ** (2. - 2) * self.f2p / self.rhobar, x=self.rho, initial=0.)
        ss4p_integral = cumtrapz(self.z ** (2. - 4) * self.f4p / self.rhobar, x=self.rho, initial=0.)
        ss6p_integral = cumtrapz(self.z ** (2. - 6) * self.f6p / self.rhobar, x=self.rho, initial=0.)
        ss8p_integral = cumtrapz(self.z ** (2. - 8) * self.f8p / self.rhobar, x=self.rho, initial=0.)

        # int_z^1 = int_0^1 - int_0^z
        ss0p_integral = ss0p_integral[-1] - ss0p_integral
        ss2p_integral = ss2p_integral[-1] - ss2p_integral
        ss4p_integral = ss4p_integral[-1] - ss4p_integral
        ss6p_integral = ss6p_integral[-1] - ss6p_integral
        ss8p_integral = ss8p_integral[-1] - ss8p_integral


        if False:
            self.ss0 = self.m / self.mtot / self.z ** 3 # (B.8)
        else:
            # this form doesn't require explicit knowledge of m or mtot.
            self.ss0 = self.rho / self.rhobar * self.f0 \
                            - 1. / self.z ** 3. * cumtrapz(self.z ** 3. / self.rhobar * self.f0, x=self.rho, initial=0.) # (B.9)

        self.ss2 = self.rho / self.rhobar * self.f2 - 1. / self.z ** (2. + 3) * ss2_integral
        self.ss4 = self.rho / self.rhobar * self.f4 - 1. / self.z ** (4. + 3) * ss4_integral
        self.ss6 = self.rho / self.rhobar * self.f6 - 1. / self.z ** (6. + 3) * ss6_integral
        self.ss8 = self.rho / self.rhobar * self.f8 - 1. / self.z ** (8. + 3) * ss8_integral

        self.ss0p = -1. * self.rho / self.rhobar * self.f0p + 1. / self.z ** (2. - 0) \
                    * (self.rho[-1] / self.rhobar * self.f0p[-1] - ss0p_integral)

        self.ss2p = -1. * self.rho / self.rhobar * self.f2p + 1. / self.z ** (2. - 2) \
                    * (self.rho[-1] / self.rhobar * self.f2p[-1] - ss2p_integral)

        self.ss4p = -1. * self.rho / self.rhobar * self.f4p + 1. / self.z ** (2. - 4) \
                    * (self.rho[-1] / self.rhobar * self.f4p[-1] - ss4p_integral)

        self.ss6p = -1. * self.rho / self.rhobar * self.f6p + 1. / self.z ** (2. - 6) \
                    * (self.rho[-1] / self.rhobar * self.f6p[-1] - ss6p_integral)

        self.ss8p = -1. * self.rho / self.rhobar * self.f8p + 1. / self.z ** (2. - 8) \
                    * (self.rho[-1] / self.rhobar * self.f8p[-1] - ss8p_integral)

    def set_f2n_f2np(self):
        '''
        N17 eqs. (B.16) and (B.17).
        '''

        self.f2 = 3. / 5 * self.s2 + 12. / 35 * self.s2 ** 2 + 6. / 175 * self.s2 ** 3 \
                    + 24. / 35 * self.s2 * self.s4 + 40. / 231 * self.s4 ** 2 \
                    + 216. / 385 * self.s2 ** 2 * self.s4 - 184. / 1925 * self.s2 ** 4

        self.f4 = 1. / 3 * self.s4 + 18. / 35 * self.s2 ** 2 + 40. / 77 * self.s2 * self.s4 \
                    + 36. / 77 * self.s2 ** 3 + 90. / 143 * self.s2 * self.s6 \
                    + 162. / 1001 * self.s4 ** 2 + 6943. / 5005 * self.s2 ** 2 * self.s4 \
                    + 486. / 5005 * self.s2 ** 4

        self.f6 = 3. / 13 * self.s6 + 120. / 143 * self.s2 * self.s4 + 72. / 143 * self.s2 ** 3 \
                    + 336. / 715 * self.s2 * self.s6 + 80. / 429 * self.s4 ** 2 \
                    + 216. / 143 * self.s2 ** 2 * self.s4 + 432. / 715 * self.s2 ** 4

        self.f8 = 3. / 17 * self.s8 + 168. / 221 * self.s2 * self.s6 + 2450. / 7293 * self.s4 ** 2 \
                    + 3780. / 2431 * self.s2 ** 2 * self.s4 + 1296. / 2431 * self.s2 ** 4

        self.f0p = 3. / 2 - 3. / 10 * self.s2 ** 2 - 2. / 35 * self.s2 ** 3 - 1. / 6 * self.s4 ** 2 \
                    - 6. / 35 * self.s2 ** 2 * self.s4 + 3. / 50 * self.s2 ** 4

        self.f2p = 3. / 5 * self.s2 - 3. / 35 * self.s2 ** 2 - 6. / 35 * self.s2 * self.s4 \
                    + 36. / 175 * self.s2 ** 3 - 10. / 231 * self.s4 ** 2 - 17. / 275 * self.s2 ** 4 \
                    + 36. / 385 * self.s2 ** 2 * self.s4

        self.f4p = 1. / 3 * self.s4 - 9. / 35 * self.s2 ** 2 - 20. / 77 * self.s2 * self.s4 \
                    - 45. / 143 * self.s2 * self.s6 - 81. / 1001 * self.s4 ** 2 + 1. / 5 * self.s2 ** 2 * self.s4
                    # f4p has an s_2**3 in Z+T. NN says it shouldn't be there (Oct 4 2017).

        self.f6p = 3. / 13 * self.s6 - 75. / 143 * self.s2 * self.s4 + 270. / 1001 * self.s2 ** 3 \
                    - 50. / 429 * self.s4 ** 2 + 810. / 1001 * self.s2 ** 2 * self.s4 - 54. / 143 * self.s2 ** 4 \
                    - 42. / 143 * self.s2 * self.s6

        self.f8p = 3. / 17 * self.s8 - 588. / 1105 * self.s2 * self.s6 - 1715. / 7293 * self.s4 ** 2 \
                    + 2352. / 2431 * self.s2 ** 2 * self.s4 - 4536. / 12155 * self.s2 ** 4

    def set_aa0(self):
        self.aa0 = (1. + 2. / 5 * self.s2 ** 2 - 4. / 105 * self.s2 ** 3 + 2. / 9 * self.s4 ** 2 \
                    + 43. / 175 * self.s2 ** 4 - 4. / 35 * self.s2 ** 2 * self.s4) * self.ss0 \
                    + (-3. / 5 * self.s2 + 12. / 35 * self.s2 ** 2 - 234. / 175 * self.s2 ** 3 \
                    + 24. / 35 * self.s2 * self.s4) * self.ss2 \
                    + (-5. / 9 * self.s4 + 6. / 7 * self.s2 ** 2) * self.ss4 \
                    + self.ss0p \
                    + (2. / 5 * self.s2 + 2. / 35 * self.s2 ** 2 + 4. / 35 * self.s2 * self.s4 \
                    - 2. / 25 * self.s2 ** 3) * self.ss2p \
                    + (4. / 9 * self.s4 + 12. / 35 * self.s2 ** 2) * self.ss4p \
                    + self.small / 3 * (1. - 2. / 5 * self.s2 - 9. / 35 * self.s2 ** 2 \
                    - 4. / 35 * self.s2 * self.s4 + 22. / 525 * self.s2 ** 3)

    def set_s2n(self, force_full=False):
        '''
        performs a root find to find the figure functions s_2n from the current S_2n, S_2n^prime, and m.
        '''

        def aa2n(s2n, k):
            """B.12-15"""
            s2, s4, s6, s8 = s2n

            aa2 = (-1. * s2 + 2. / 7 * s2 ** 2 + 4. / 7 * s2 * s4 - 29. / 35 * s2 ** 3 + 100. / 693 * s4 ** 2 \
                    + 454. / 1155 * s2 ** 4 - 36. / 77 * s2 ** 2 * s4) * self.ss0[k] \
                    + (1. - 6. / 7 * s2 - 6. / 7 * s4 + 111. / 35 * s2 ** 2 - 1242. / 385 * s2 ** 3 + 144. / 77 * s2 * s4) * self.ss2[k] \
                    + (-10. / 7 * s2 - 500. / 693 * s4 + 180. / 77 * s2 ** 2) * self.ss4[k] \
                    + (1. + 4. / 7 * s2 + 1. / 35 * s2 ** 2 + 4. / 7 * s4 - 16. / 105 * s2 ** 3 + 24. / 77 * s2 * s4) * self.ss2p[k] \
                    + (8. / 7 * s2 + 72. / 77 * s2 ** 2 + 400. / 693 * s4) * self.ss4p[k] \
                    + self.small / 3 * (-1. + 10. / 7 * s2 + 9. / 35 * s2 ** 2 - 4. / 7 * s4 + 20./ 77 * s2 * s4 - 26. / 105 * s2 ** 3)

            aa4 = (-1. * s4 + 18. / 35 * s2 ** 2 - 108. / 385 * s2 ** 3 + 40. / 77 * s2 * s4 + 90. / 143 * s2 * s6 + 162. / 1001 * s4 ** 2 \
                    + 16902. / 25025 * s2 ** 4 - 7369. / 5005 * s2 ** 2 * s4) * self.ss0[k] \
                    + (-54. / 35 * s2 - 60. / 77 * s4 + 648. / 385 * s2 ** 2 \
                    - 135. / 143 * s6 + 21468. / 5005 * s2 * s4 - 122688. / 25025 * s2 ** 3) * self.ss2[k] \
                    + (1. - 100. / 77 * s2 - 810. / 1001 * s4 + 6368. / 1001 * s2 ** 2) * self.ss4[k] \
                    - 315. / 143 * s2 * self.ss6[k] \
                    + (36. / 35 * s2 + 108. / 385 * s2 ** 2 + 40. / 77 * s4 + 3578. / 5005 * s2 * s4 \
                    - 36. / 175 * s2 ** 3 + 90. / 143 * s6) * self.ss2p[k] \
                    + (1. + 80. / 77 * s2 + 1346. / 1001 * s2 ** 2 + 648. / 1001 * s4) * self.ss4p[k] \
                    + 270. / 143 * s2 * self.ss6p[k] \
                    + self.small / 3 * (-36. / 35 * s2 + 114. / 77 * s4 + 18. / 77 * s2 ** 2 \
                    - 978. / 5005 * s2 * s4 + 36. / 175 * s2 ** 3 - 90. / 143 * s6)

            aa6 = (-s6 + 10. / 11 * s2 * s4 - 18. / 77 * s2 ** 3 + 28. / 55 * s2 * s6 + 72. / 385 * s2 ** 4 + 20. / 99 * s4 ** 2 \
                    - 54. / 77 * s2 ** 2 * s4) * self.ss0[k] \
                    + (-15. / 11 * s4 + 108. / 77 * s2 ** 2 - 42. / 55 * s6 - 144. / 77 * s2 ** 3 + 216. / 77 * s2 * s4) * self.ss2[k] \
                    + (-25. / 11 * s2 - 100. / 99 * s4 + 270. / 77 * s2 ** 2) * self.ss4[k] \
                    + (1. - 98. / 55 * s2) * self.ss6[k] \
                    + (10. / 11 * s4 + 18. / 77 * s2 ** 2 + 36. / 77 * s2 * s4 + 28. / 55 * s6) * self.ss2p[k] \
                    + (20. / 11 * s2 + 108. / 77 * s2 ** 2 + 80. / 99 * s4) * self.ss4p[k] \
                    + (1. + 84. / 55 * s2) * self.ss6p[k] \
                    + self.small / 3 * (-10. / 11 * s4 - 18. / 77 * s2 ** 2 + 34. / 77 * s2 * s4 + 82. / 55 * s6)

            aa8 = (-s8 + 56. / 65 * s2 * s6 + 72. / 715 * s2 ** 4 + 490. / 1287 * s4 ** 2 - 84. / 143 * s2 ** 2 * s4) * self.ss0[k] \
                    + (-84. / 65 * s6 - 144. / 143 * s2 ** 3 + 336. / 143 * s2 * s4) * self.ss2[k] \
                    + (-2450. / 1287 * s4 + 420. / 143 * s2 ** 2) * self.ss4[k] \
                    - 196. / 65 * s2 * self.ss6[k] \
                    + self.ss8[k] \
                    + (56. / 65 * s6 + 56. / 143 * s2 * s4) * self.ss2p[k] \
                    + (1960. / 1287 * s4 + 168. / 143 * s2 ** 2) * self.ss4p[k] \
                    + 168. / 65 * s2 * self.ss6p[k] \
                    + self.ss8p[k] \
                    + self.small / 3 * (-56. / 65 * s6 - 56. / 143 * s2 * s4)


            return np.array([aa2, aa4, aa6, aa8])

        if self.method_for_aa2n_solve == 'full' or force_full: # default

            for k in np.arange(self.nz):
                s2n = np.array([self.s2[k], self.s4[k], self.s6[k], self.s8[k]])
                sol = root(aa2n, s2n, args=(k)) # increasing xtol has essentially no effect
                if not sol['success']:
                    print (sol)
                    raise RuntimeError('failed in solve for s_2n.')

                # store solution
                self.s2[k], self.s4[k], self.s6[k], self.s8[k] = sol.x

                # store residuals for diagnostics
                # self.aa2[k], self.aa4[k], self.aa6[k], self.aa8[k] = aa2n(sol.x, k)

        elif 'cubic' in self.method_for_aa2n_solve:
            try:
                fskip = int(self.method_for_aa2n_solve.split()[1])
            except IndexError:
                fskip = 10
            nskip = int(self.nz / fskip)

            for k in np.arange(self.nz)[::nskip]:
                s2n = np.array([self.s2[k], self.s4[k], self.s6[k], self.s8[k]])
                # sol = root(aa2n, s2n, args=(k)
                sol = root(aa2n, s2n, args=(k), options={'xtol':1e-4})
                if not sol['success']:
                    print (sol)
                    raise RuntimeError('failed in solve for s_2n.')

                # store solution
                self.s2[k], self.s4[k], self.s6[k], self.s8[k] = sol.x

            tck2 = splrep(self.l[::nskip], self.s2[::nskip], k=3)
            tck4 = splrep(self.l[::nskip], self.s4[::nskip], k=3)
            tck6 = splrep(self.l[::nskip], self.s6[::nskip], k=3)
            tck8 = splrep(self.l[::nskip], self.s8[::nskip], k=3)

            self.s2 = splev(self.l, tck2)
            self.s4 = splev(self.l, tck4)
            self.s6 = splev(self.l, tck6)
            self.s8 = splev(self.l, tck8)

        else:
            raise ValueError('unable to parse aa2n solve method %s' % self.method_for_aa2n_solve)

        # note misprinted power on first term in Z+T (28.12)
        self.s0 = - 1. / 5 * self.s2 ** 2 \
                    - 2. / 105 * self.s2 ** 3 \
                    - 1. / 9 * self.s4 ** 2 \
                    - 2. / 35 * self.s2 ** 2 * self.s4

        return

    def set_j2n(self):
        '''
        J_2n :: the harmonic coefficients. eq. (B.11)
        '''

        self.j2n_old = self.j2n

        self.j2 = - 1. * (self.rm / self.r_eq[-1]) ** 2. * self.ss2[-1]
        self.j4 = - 1. * (self.rm / self.r_eq[-1]) ** 4. * self.ss4[-1]
        self.j6 = - 1. * (self.rm / self.r_eq[-1]) ** 6. * self.ss6[-1]
        self.j8 = - 1. * (self.rm / self.r_eq[-1]) ** 8. * self.ss8[-1]

        self.j0 = - self.ss0[-1]

        self.j2n = np.array([self.j2, self.j4, self.j6, self.j8])

        self.dj2n = self.j2n - self.j2n_old

        return

    def set_j2n_radau(self):
        # for comparison, calculate the first-order ellipticity from Clairaut theory w/ the
        # Radau-Darwin approximation
        etanum = cumtrapz(self.rho * self.l ** 4, x=self.l, initial=1)
        etaden = cumtrapz(self.rho * self.l ** 2, x=self.l, initial=1)
        eta = 25. / 4 * (1. - etanum / etaden / self. l ** 2) ** 2 - 1.
        k0 = np.where(self.l / self.l[-1] > 0.1)[0][0]
        eta[:k0] = eta[k0] # flatten out the near-origin points
        # the chicanery is to get the integral from rp=r to rp=a after first calculating the integral from rp=0 to rp=r
        integral = cumtrapz(eta / self.l, x=self.l, initial=0.)
        integral -= integral[-1]
        eps = np.exp(integral) # relative ellipticity; unity at surface
        qq = trapz(self.rho * self.l ** 4 * eps * (5. + eta), x=self.l)
        mtot = self.mtot_calc if hasattr(self, 'mtot_calc') else self.mtot
        qq *= np.pi * 4 / 5 / mtot / self.rm ** 2 # note factor of 2 typo in Fuller (A.3)
        eps *= self.small / 3 / (1. - qq) # true ellipticity

        self.j2_radau = qq * eps[-1]
        self.eps_radau = eps
        self.req_radau = self.l[-1] * (1. + 0.5 * self.eps_radau[-1]) # p2 = 1 / 2 * (3x^2 - 1)
        # self.rpol_radau = self.l[-1] * (1. - 0.5 * self.eps_radau[-1]) # p2 = 1 / 2 * (3x^2 - 1)

        self.j2n_old = self.j2n

        self.j2 = self.j2_radau
        self.j4 = 0.
        self.j6 = 0.
        self.j8 = 0.

        self.j2n = np.array([self.j2, self.j4, self.j6, self.j8])
        self.dj2n = self.j2n - self.j2n_old

        self.z = self.l / self.rm

        self.f0p = 3. / 2
        ss0p_integral = cumtrapz(self.z ** (2. - 0) * self.f0p / self.rhobar, x=self.rho, initial=0.)
        ss0p_integral = ss0p_integral[-1] - ss0p_integral
        self.ss0 = self.rho / self.rhobar * self.f0 \
                        - 1. / self.z ** 3. * cumtrapz(self.z ** 3. / self.rhobar * self.f0, x=self.rho, initial=0.) # (B.9)
        self.ss0p = -1. * self.rho / self.rhobar * self.f0p + 1. / self.z ** (2. - 0) \
                    * (self.rho[-1] / self.rhobar * self.f0p - ss0p_integral)

if __name__ == '__main__':

    params = {}
    # here initializing values roughly like saturn
    params['small'] = 0.142
    params['mtot'] = 5.6834e29
    params['req'] = 60268e5
    params['t1'] = 140.
    
    # model atmospheres just because ongp requires them; shouldn't affect anything
    params['atm_option'] = 'f11_tables'
    params['atm_planet'] = 'sat'
    
    # eos choices
    params['z_eos_option'] = 'aneos mix'
    params['hhe_eos_option'] = 'mh13_scvh'
    params['ice_to_rock'] = 0.5 # water ice to rock (serpentine) mass fraction; only applies if z_eos_option is aneos mix
    
    # composition choices
    if False: # make a three-layer homogeneous model: core, inner envelope, outer envelope
        params['model_type'] = 'three_layer' # or cosine_yz, or linear_yz
        params['y1'] = 0.12 # outer envelope helium mass fraction
        params['mean_y_xy'] = 0.275 # M_He / (M_H + M_He); outer iterations will adjust y2 to satisfy this
        params['z1'] = 0.02 # outer envelope heavy element mass fraction
        params['z2'] = 0.3 # inner envelope heavy element mass fraction
        params['mcore'] = 10. # initial guess for core mass (earth masses); will be adjusted in iterations to satisfy correct total mass
        params['pt'] = 2. # pressure (Mbar) corresponding to inner/envelope envelope boundary
    else: # make a model with sinusoid Y and Z profiles (single stable region)
        params['model_type'] = 'cosine_yz' # can also try linear_yz which works similarly
        params['y2_xy'] = 0.95 # y/(x+y) in inner core
        params['mean_y_xy'] = 0.275 # M_He / (M_H + M_He); outer iterations will adjust y1_xy to satisfy this
        params['y1_xy'] = 0.1 # initial guess for y/(x+y) in outer envelope; will be adjusted in iterations to satisfy specified mean_y_xy
        params['z1'] = 0.02 # outer envelope heavy element mass fraction
        params['z2'] = 0.9 # inner core heavy element mass fraction
        params['c'] = 0.1 # initial guess for centroid radius of stable region; will be adjusted in iterations to satisfy correct total mass
        params['w'] = 0.2 # radial width of stable region
        # params['w']
        
    t = tof4(params)
    t.initialize_model()
    t.initialize_tof_vectors()
    t.relax()
    