# a fourth-order theory of figures based on
# the summmary in appendix B of Nettelmann (2017, A&A 606, A139)
# doi:10.1051/0004-6361/201731550

import numpy as np
import const
from scipy.optimize import root
from scipy.integrate import trapz, cumtrapz, solve_ivp
from scipy.interpolate import splrep, splev, interp1d
from scipy.special import legendre
import time
import os
# import ongp

class ConvergenceError(Exception):
    pass

class ToFAdjustError(Exception):
    pass

class EOSError(Exception):
    pass

class UnphysicalParameterError(Exception):
    pass

class model_container:
    '''
    container for quantities not fundamental to the tof4 model, e.g., composition, etc.
    '''
    pass

class tof4:
    def __init__(self, hhe_eos, z_eos, tof_params={}):

        # create a blank model instance to hold info tof doesn't care about directly, like composition and temperature
        self.model = model_container()
        # these should be eos instances that each have a 'get' method that takes (logp, logt, {composition parameters})
        # and returns a dictionary of eos results like logrho, grada, gamma1, etc.
        self.model.hhe_eos = hhe_eos
        self.model.z_eos = z_eos

        # parse contents of the params dictionary
        self.max_iters_outer = tof_params['max_iters_outer'] if 'max_iters_outer' in tof_params else 100
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
        # flag to adjust "smallness parameter" (squared spin divided by GM/Rmean^3) to preserve actual spin frequency as mean radius changes during iterations
        self.adjust_small = tof_params['adjust_small'] if 'adjust_small' in tof_params else True

        if not 'save_vector_output' in tof_params.keys():
            tof_params['save_vector_output'] = True

        self.params = tof_params

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

        self.mtot = self.params['mtot'] # important, sets target total mass throughout iterations and enters mean density
        self.rtot = self.params['req'] * (1. - self.small ** 2) # rough idea of total mean radius; only used to get rough initial model

        self.model.t1 = self.params['t1']
        self.model.f_ice = self.params['f_ice']
        self.model.y = np.zeros(self.nz)
        self.model.z = np.zeros(self.nz)
        self.model.p = np.ones(self.nz) * 1e12
        self.model.t = np.ones_like(self.model.p) * 1e4

        if 'use_gauss_lobatto' in self.params and self.params['use_gauss_lobatto']:
            # here we want a prescribed fraction radius versus zone number, so we'll set this
            # and iterate to get roughly the correct total mass for our starting model
            def gauss_lobatto(N,xmin,xmax):
                '''
                Constructs Gauss-Lobatto grid and 1st/2nd order spectral
                derivative matrices appropriate for Chebyshev collocation
                '''
                # preliminaries
                M=N+1
                x = np.linspace(0,N,M)
                x = np.cos(np.pi*x/N)
                # map to physical domain x\in[xmin,xmax]
                x   = (-x+1.)*(xmax-xmin)/2. + xmin
                return x
            self.model.r = gauss_lobatto(self.nz-1, 0., 1.) * self.rtot
            self.model.dr = np.gradient(self.model.r)
            t0 = time.time()
            z = 0.1
            # iterate for a very rough 1d starting model
            for i in np.arange(2):
                hhe_res = self.model.hhe_eos.get(np.log10(self.model.p), np.log10(self.model.t), 0.275 * (1. - z))
                z_res = self.model.z_eos.get(np.log10(self.model.p), np.log10(self.model.t), self.model.f_ice)
                rhoinv = z / 10 ** z_res['logrho'] + (1. - z) / 10 ** hhe_res['logrho']
                self.model.rho = rhoinv ** -1. # 10 ** hhe_res['logrho']
                # integrate mass conservation to update mass
                self.model.m = cumtrapz(np.pi * 4. * self.model.r ** 2 * self.model.rho, x=self.model.r, initial=0.)
                self.model.m[0] = np.pi * 4. * self.model.rho[0] * self.model.r[0] ** 3 / 3.
                self.model.dm = np.diff(self.model.m)
                # integrate hydrostatic balance to update pressure
                dp = const.cgrav * self.model.m[1:] * self.model.dm / 4. / np.pi / self.model.r[1:] ** 4
                self.model.p[-1] = 1e6 # 1 bar
                self.model.p[:-1] = 1e6 + np.cumsum(dp[::-1])[::-1]
                # integrate adiabatic temperature gradient to update temperature
                self.model.grada = hhe_res['grada']
                interp_grada = interp1d(self.model.p, self.model.grada, fill_value='extrapolate') # interp1d(self.p, self.grada)
                def dtdp(p, t):
                    # print(f'{p:e} {self.model.p[0]:e} {self.model.p[0]-p:e}')
                    return t / p * interp_grada(p)
                p_eval = self.model.p[::-1]
                assert not np.any(np.isnan(self.model.p))
                assert not np.any(self.model.p[1:] > self.model.p[0])
                assert not np.any(self.model.p[:-1] < self.model.p[-1])
                assert self.model.p[-1] == 1e6
                assert interp_grada(p_eval[0])
                assert interp_grada(p_eval[-1])
                sol = solve_ivp(dtdp, (p_eval[0], p_eval[-1]), np.array([self.model.t1]), t_eval=p_eval)
                assert sol.success, 'failed in integrate_temperature'
                self.model.t = sol.y[0][::-1]
                # check total mass relative to target and adjust the trial value of z to get closer next pass
                mtot_relerr = (self.mtot - self.model.m[-1]) / self.mtot
                if self.verbosity > 0:
                    print(f'{self.model.t[0]:10e} {z:6.3f} {self.model.m[-1]:10e} {mtot_relerr:10e} {time.time() - t0:10f}')
                z *= np.exp(mtot_relerr)
        else:
            # default mesh is a prescribed fractional mass versus zone number; iterate to get roughly correct radius
            import mesh
            self.model.m = mesh.mesh(self.nz).mesh_func * self.mtot
            self.model.dm = np.diff(self.model.m)
            q = np.zeros_like(self.model.p)
            if self.verbosity > 0:
                print('create initial model')
                print(f"{'tcenter':>10} {'rtot':>10} {'relerr':>10} {'et_s':>10}")
            t0 = time.time()
            z = 0.1
            for i in np.arange(2):

                hhe_res = self.model.hhe_eos.get(np.log10(self.model.p), np.log10(self.model.t), 0.275 * (1. - z))
                z_res = self.model.z_eos.get(np.log10(self.model.p), np.log10(self.model.t), self.model.f_ice)
                rhoinv = z / 10 ** z_res['logrho'] + (1. - z) / 10 ** hhe_res['logrho']
                self.model.rho = rhoinv ** -1. # 10 ** hhe_res['logrho']
                # integrate mass conservation to update radii
                q[1:] = 3. * self.model.dm / 4 / np.pi / self.model.rho[1:]
                self.model.r = np.cumsum(q) ** (1. / 3)
                # integrate hydrostatic balance to update pressure
                dp = const.cgrav * self.model.m[1:] * self.model.dm / 4. / np.pi / self.model.r[1:] ** 4
                self.model.p[-1] = 1e6 # 1 bar
                self.model.p[:-1] = 1e6 + np.cumsum(dp[::-1])[::-1]
                # integrate adiabatic temperature gradient to update temperature
                self.model.grada = hhe_res['grada']
                interp_grada = interp1d(self.model.p, self.model.grada, fill_value='extrapolate') # interp1d(self.p, self.grada)
                def dtdp(p, t):
                    # print(f'{p:e} {self.model.p[0]:e} {self.model.p[0]-p:e}')
                    return t / p * interp_grada(p)
                p_eval = self.model.p[::-1]
                assert not np.any(np.isnan(self.model.p))
                assert not np.any(self.model.p[1:] > self.model.p[0])
                assert not np.any(self.model.p[:-1] < self.model.p[-1])
                assert self.model.p[-1] == 1e6
                assert interp_grada(p_eval[0])
                assert interp_grada(p_eval[-1])
                sol = solve_ivp(dtdp, (p_eval[0], p_eval[-1]), np.array([self.model.t1]), t_eval=p_eval)
                assert sol.success, 'failed in integrate_temperature'
                self.model.t = sol.y[0][::-1]
                # check total radius relative to target and adjust the trial value of z to get closer next pass
                rtot_relerr = (self.rtot - self.model.r[-1]) / self.rtot
                if self.verbosity > 0:
                    print(f'{self.model.t[0]:10e} {self.model.r[-1]:10e} {rtot_relerr:10e} {time.time() - t0:10f}')
                z /= np.exp(rtot_relerr)

        # assert False

        if np.any(np.isnan(self.model.p)):
            nnan = len(self.model.p[np.isnan(self.model.p)])
            raise EOSError(f'{nnan} nans in pressure after integrate hydrostatic on iteration {self.static_iters}.')

        for key in 'z1', 'z2',:
            if key in self.params: setattr(self.model, key, self.params[key])

        if self.params['model_type'] == 'three_layer':
            self.model.mcore = self.params['mcore']
            self.model.y1 = self.params['y1']
            self.model.kcore = np.where(self.model.m > self.model.mcore * const.mearth)[0][0]
            self.model.ptrans = self.params['pt']
        elif self.params['model_type'] == 'continuous':

            if 'c1' in self.params: # dual cavity model
                assert 'w' not in self.params
                assert 'c' not in self.params
                self.model.c1 = self.params['c1']
                self.model.c2 = self.params['c2']
                self.model.w1 = self.params['w1']
                self.model.w2 = self.params['w2']

            elif 'rstab' in self.params: # single cavity model, cavity extending out to r=rstab from either r=rstab_in or r=0 if rstab_in is not defined
                assert 'w' not in self.params
                assert 'c' not in self.params
                self.model.rstab = self.params['rstab']
                if 'rstab_in' in self.params:
                    self.model.rstab_in = self.params['rstab_in']

            else: # standard single cavity model: work with gradient centroid and width
                assert 'c2' not in self.params
                assert 'c1' not in self.params
                assert 'w1' not in self.params
                assert 'w2' not in self.params

                self.model.c = self.params['c'] # y/z gradient centroid
                self.model.w = self.params['w'] # y/z gradient full-width

            self.model.y1_xy = self.params['y1_xy'] # y/(x+y) of uniform outer envelope
            try:
                self.model.y2_xy = self.params['y2_xy'] # y/(x+y) of "helium shell"
            except KeyError:
                self.model.y2_xy = None # adjust will catch this and set y2_xy=y1_xy
        else:
            raise ValueError(f"model_type {self.params['model_type']} not recognized")

        # copy over primary structure quantities for tof iterations
        self.l = self.model.r
        self.rho = self.model.rho
        self.p = self.model.p

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
                    raise UnphysicalParameterError('model wants negative core mass')

                self.adjust_three_layer()

            elif self.params['model_type'] == 'continuous':

                self.model.mhe = trapz(self.model.y, x=self.m_calc)
                self.model.mz = trapz(self.model.z, x=self.m_calc)
                self.model.ymean_xy = self.model.mhe / (self.mtot_calc - self.model.mz) # current actual M_He / (M_H + M_He)

                # how badly are we doing with total mass and mean y_xy?
                mtot_relerr = (self.mtot - self.mtot_calc) / self.mtot
                ymean_relerr = (self.params['ymean_xy'] - self.model.ymean_xy) / self.params['ymean_xy']

                # based on the parameters that have been passed, choose which quantity to adjust to get closer to desired total mass
                if 'c1' in self.params: # dual cavity model; adjust c2
                    adjust_qty = 'c2'
                elif 'c' in self.params: # single cavity model; adjust c
                    adjust_qty = 'c'
                elif 'rstab' in self.params:
                    adjust_qty = 'z2' # now preferred
                    # adjust_qty = 'z1' # seemed like a better handle but the strong covariance w/ y1_xy (varied to satisfy desired ymean) leads to convergence woes
                    # adjust_qty = 'rstab_in' # bad

                    # give user the option to override this, e.g., for specific plots
                    if 'force_adjust_z1' in self.params and self.params['force_adjust_z1']:
                        adjust_qty = 'z1'

                else:
                    raise ToFAdjustError('unclear which quantity to adjust to get closer to desired total mass.')

                # not clear if this extra tracking will be necessary but it might if we run into small y1_xy values
                if not hasattr(self, 'y1_xy_prev'): self.y1_xy_prev = np.array([])
                if not hasattr(self, 'ymean_xy_prev'): self.ymean_xy_prev = np.array([])
                if not hasattr(self, 'c_prev'): self.c_prev = np.array([])
                if not hasattr(self, 'mtot_calc_prev'): self.mtot_calc_prev = np.array([])
                self.y1_xy_prev = np.append(self.y1_xy_prev, self.model.y1_xy)
                self.ymean_xy_prev = np.append(self.ymean_xy_prev, self.model.ymean_xy)
                self.c_prev = np.append(self.c_prev, getattr(self.model, adjust_qty))
                self.mtot_calc_prev = np.append(self.mtot_calc_prev, self.mtot_calc)


                setattr(self.model, adjust_qty, getattr(self.model, adjust_qty) + mtot_relerr)
                # self.model.c += mtot_relerr

                # this might fail if y1_xy is close to zero
                f = 1.
                self.model.y1_xy *= (1. + f * ymean_relerr)

                # sanity checks to make sure values make sense, and give up if needed
                if getattr(self.model, adjust_qty) < 0. and self.outer_iteration > 20:
                    raise ToFAdjustError(f'model wants {adjust_qty}={getattr(self.model, adjust_qty)} < 0.')
                elif getattr(self.model, adjust_qty) > 1.:
                    if self.outer_iteration < 20:
                        # early in iterations; ease toward unity in case a good solution is in there somewhere
                        setattr(self.model, adjust_qty, 0.5 *(self.c_prev[-1] + 1.))
                    else:
                        raise ToFAdjustError(f'model wants {adjust_qty}={getattr(self.model, adjust_qty)} > 1. model might have an unrealistically low total z content.')
                elif getattr(self.model, adjust_qty) < 0.:
                    # don't let it go negative, ease it toward zero in hopes that a converged model exists with this quantity non-negative
                    setattr(self.model, adjust_qty, 0.5 * self.c_prev[-1])

                if self.model.y1_xy < 0. and self.outer_iteration > 20:
                    raise ToFAdjustError(f'model wants y1_xy={self.model.y1_xy} < 0.')
                elif self.model.y1_xy < 0.:
                    self.model.y1_xy = 0.5 * self.y1_xy_prev[-1]
                # elif self.model.y1_xy < 2e-2 and self.outer_iteration > 20:
                    # raise ToFAdjustError(f'model wants y1_xy < 1e-2 after 20 iterations; convergence is unlikely')

                self.adjust() # enforce composition profile for current global parameters, query eos, integrate temperatures

                if 'debug_adjust' in self.params and self.params['debug_adjust']:
                    import pickle
                    debug = {'y_xy':self.model.y_xy, 'ymean_xy':self.model.ymean_xy, 'y1_xy':self.model.y1_xy, 'z':self.model.z, 'm':self.m_calc, adjust_qty:getattr(self.model, adjust_qty), 'mtot_calc':self.mtot_calc}
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
            adjust_output = {}
            if self.params['model_type'] == 'three_layer':
                adjust_output = {
                    'mcore':self.model.mcore,
                    'y1':self.model.y1,
                    'y2':self.model.y2,
                    'ymean_xy':self.model.ymean_xy if hasattr(self.model, 'ymean_xy') else -1,
                    }
            elif self.params['model_type'] == 'continuous':
                adjust_output = {
                    # 'c':self.model.c,
                    'y1_xy':self.model.y1_xy if hasattr(self.model, 'y1_xy') else -1,
                    'ymean_xy':self.model.ymean_xy if hasattr(self.model, 'ymean_xy') else -1,
                    }
                # if hasattr(self.model, 'c2'):
                #     adjust_output['c2'] = self.model.c2
                # elif hasattr(self.model, 'c'):
                #     adjust_output['c'] = self.model.c
                # elif hasattr(self.model, 'rstab'):
                #     adjust_output['z1'] = self.model.z1
                # else:
                #     raise ValueError('unclear what quantity will be adjusted to satisfy total mass.')
                adjust_output[adjust_qty] = getattr(self.model, adjust_qty)
            else:
                raise ValueError(f"model_type {self.params['model_type']} not recognized")

            if self.verbosity > 1 or (self.outer_iteration == 0 and self.verbosity > 0):
                # names = 'iout', 'iin', 'nz', 'rhobar', \
                #     # 'mtot_calc', \
                #     'dmtot', \
                #     *list(adjust_output), \
                #     'z2', 'z1', \
                #     'et_total', 'j2', 'j4', 'j6', 'req', 'rm', 'rpol', 'small'
                # print (('%5s ' * 2 + '%11s ' * (len(names) - 2)) % names)
                print(f"{'iout':>5}", end=' ')
                print(f"{'iin':>5}", end=' ')
                print(f"{'nz':>7}", end=' ')
                for name in 'rhobar', 'log_dmtot':
                    print(f"{name:>11}", end=' ')
                for key in adjust_output:
                    print(f"{key:>11}", end=' ')
                for name in 'z2', 'z1', 'et_tot', 'j2', 'j4', 'j6', 'req', 'rm', 'rpol', 'small':
                    print(f"{name:>11}", end=' ')
                print()

            if self.verbosity > 0:
                # data = self.outer_iteration, self.inner_iteration, self.rhobar, \
                #         # self.mtot_calc / self.mtot, \
                #         np.log10(np.abs(1. - self.mtot_calc / self.mtot)), \
                #         *[adjust_output[key] for key in list(adjust_output)], \
                #         self.model.z2 if hasattr(self.model, 'z2') else -1, \
                #         self.model.z1, self.nz, \
                #         ,
                # print (('%5i ' * 2 + '%11.5g ' * (len(data) - 2)) % data)
                print(f'{self.outer_iteration:5}', end=' ')
                print(f'{self.inner_iteration:5}', end=' ')
                print(f'{self.nz:7}', end=' ')
                print(f'{self.rhobar:11.5g}', end=' ')
                print(f'{np.log10(np.abs(1. - self.mtot_calc / self.mtot)):11.2e}', end=' ')
                for key, val in adjust_output.items():
                    print(f'{val:11.5g}', end=' ')
                print(f"{self.model.z2 if hasattr(self.model, 'z2') else -1:11.5g}", end=' ')
                print(f'{self.model.z1:11.5g}', end=' ')
                for val in self.et_total, self.j2, self.j4, self.j6, self.r_eq[-1], self.rm, self.r_pol[-1], self.small:
                    print(f'{val:11.5e}', end=' ')
                print()

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

            if self.adjust_small:
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
                # print(self.j2n_rtol, (self.j2n_last_outer - self.j2n) / (self.j2n+1e-20))
                # print(self.mtot_rtol, abs(self.mtot_calc - self.mtot) / self.mtot)
                # print(self.ymean_rtol, abs(self.model.ymean_xy - self.params['ymean_xy']) / self.params['ymean_xy'])
                if np.all(abs((self.j2n_last_outer - self.j2n) / (self.j2n+1e-20)) < self.j2n_rtol) \
                    and self.outer_iteration > 10 \
                    and self.inner_done \
                    and abs(self.mtot_calc - self.mtot) / self.mtot < self.mtot_rtol \
                    and abs(self.model.ymean_xy - self.params['ymean_xy']) / self.params['ymean_xy'] < self.ymean_rtol:

                    if self.verbosity >= 3.:
                        print ('terminate outer loop; all dj2n/j2n < %g.' % self.j2n_rtol)

                    self.outer_done = True

                    t0 = time.time()
                    self.set_s2n(force_full=False) # this is actually slow if force_full==True; effect on final shape functions appears negligible
                    # if self.verbosity > 0: print(f'final set_s2n completed in {time.time()-t0} s')

                    self.save_model() # calculate auxiliary quantities of interest and write the thing to disk

                    break

            self.j2n_last_outer = self.j2n

        return

    def adjust_three_layer(self):
        '''enforce new mcore, set y and z, reintegrate *adiabatic* t profile, and recalculate densities.
        this will not in general conserve the total mass of the model; the outer loop itself handles this by
        comparing mtot to mtot_calc and telling this routine what mcore to set.'''

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
        self.model.y[kcore:] = self.tof_params['y1'] # 0.27
        # y = self.model.equilibrium_y_profile({'phase_t_offset':self.tof_params['phase_t_offset']})
        # for the ToF calculation hold y1 fixed instead of phase_t_offset
        if 'rainout_y1' in list(self.tof_params) and self.tof_params['rainout_y1'] is not None:
            assert not hasattr(self.model, 'y2')
            # y = self.model.y
            y = self.model.equilibrium_y_profile_y1(self.tof_params['rainout_y1'])
        elif 'dt' in list(self.tof_params):
            # if not hasattr(self.model, 'mhe'): self.model.mhe = np.dot(self.model.y[kcore:], self.model.dm[kcore:])
            menv = self.mtot_calc - self.model.mcore * const.mearth
            # self.model.mhe = self.tof_params['ym'] * menv
            if not hasattr(self.model, 'mhe'):
                self.model.mhe = 0.27 * menv
            self.model.ym = self.model.mhe / menv # current mean envelope helium abundance
            # print('before rainout: mhe = {} me; envelope ym = {}'.format(self.model.mhe / const.mearth, self.model.ym))
            y = self.model.equilibrium_y_profile(self.tof_params['dt'])
            self.model.y = y
            # print(' after rainout: mhe = {} me; envelope ym = {}'.format(np.dot(self.model.dm, self.model.y) / const.mearth, np.dot(self.model.dm, self.model.y) / menv))
            # print(np.dot(self.model.dm, y) / (self.mtot_calc - self.model.mcore)) # current mean envelope helium abundance
            self.model.y1 = y[-1]
            self.model.y1_xy = y[-1] / (1. - self.model.z[-1])
        elif 'hhe_phase_diagram' not in list(self.evol_params) or self.evol_params['hhe_phase_diagram'] is None:
            # no phase diagram defined
            if hasattr(self.model, 'y2') and self.model.y2:
                # doing three-layer y
                y[kcore:ktrans] = self.model.y2
                self.model.y1 = self.tof_params['y1']
                y[ktrans:] = self.model.y1
            else:
                # either doing single envelope y, or doing three-layer but self.adjust hasnt been called yet
                # whole envelope has y=ym
                # self.model.y1 = self.model.ym
                self.model.y2 = self.model.ym
                y[kcore:] = self.model.ym
        else:
            y = self.model.y
        assert not np.any(y[kcore:] == 0)
        # self.model.k_shell_top now set; compare to ptrans

        t[-1] = self.model.t1

        try:
            assert not np.any(np.isnan(self.p)), 'nans in p during adjust mcore, before t integration'
            assert not np.any(np.isnan(t)), 'nans in t during adjust mcore, before t integration'
            assert not np.any(np.isnan(self.model.grada)), 'nans in grada during adjust mcore, before t integration'
        except AssertionError as e:
            raise EOSError(e.args[0])

        self.model.gradt[kcore:] = self.model.grada[kcore:] \
            = self.model.hhe_eos.get_grada(np.log10(self.p[kcore:]), np.log10(self.model.t[kcore:]), y[kcore:])
        self.model.gradt[:kcore] = 0.
        self.model.grada[:kcore] = 0.

        if np.any(np.isnan(self.model.grada)):
            raise ValueError('nans in grada: y1={} y2={} z1={} z2={}'.format(self.model.y1, self.model.y2, self.model.z1, self.model.z2))

        if 'alternate_t_integral' in list(self.tof_params) and self.tof_params['alternate_t_integral']:
            # integrate on alternate pressure grid to improve accuracy
            from scipy.interpolate import interp1d
            p_grid = np.logspace(6, np.log10(self.p[0]), len(self.p))[::-1]
            grada_grid = interp1d(self.p, self.model.grada, fill_value='extrapolate', kind='cubic')(p_grid)
            t = np.exp(cumtrapz(grada_grid[::-1] / p_grid[::-1], x=p_grid[::-1], initial=0.))[::-1]
            t *= self.model.t1
            t = interp1d(p_grid, t, fill_value='extrapolate', kind='cubic')(self.p)
        else:
            t = np.exp(cumtrapz(self.model.gradt[::-1] / self.p[::-1], x=self.p[::-1], initial=0.))[::-1]
            t *= self.model.t1

        self.rho[:kcore] = self.model.get_rho_z(np.log10(self.p[:kcore]), np.log10(t[:kcore]))
        self.rho[kcore:] = self.model.get_rho_xyz(np.log10(self.p[kcore:]), np.log10(t[kcore:]), y[kcore:], z[kcore:])

        self.model.t = t
        self.model.y = y
        self.model.z = z

        self.model.mhe = np.dot(self.model.dm, self.model.y)
        self.model.mz = np.dot(self.model.dm, self.model.z)
        self.model.ymean_xy = self.model.mhe / (self.mtot_calc - self.model.mz) # current actual M_He / (M_H + M_He)
        self.model.ym = self.model.mhe / self.mtot_calc # current actual M_He / M_tot

        # adjust y2 to approach correct ymean
        relative_mismatch = (self.params['ymean_xy'] - self.model.ymean_xy) / self.params['ymean_xy']
        if not hasattr(self.model, 'y2'): self.model.y2 = self.model.y1
        self.model.y2 *= (1. + relative_mismatch)
        if self.model.y2 < 0: raise ToFAdjustError('got y2<0 trying to adjust y2')

        return

    def adjust(self):
        '''
        for current pressure-mass-radius profile, enforce composition profiles corresponding to the user-supplied composition params.
        then do an eos call and integrate to get temperature profile, and update the densities.
        this version is similar to the original, adjust_three_layer, but for a model with continuous Y and Z distributions.
        here Z and Y_XY each increase sinusoidally or linearly from r/R = c+w/2 down to c-w/2.
        '''

        if np.any(np.diff(self.l) <= 0.):
            raise ValueError('radius not monotone increasing')

        rf = self.l / self.l[-1] # fractional radius

        if hasattr(self.model, 'c2') or hasattr(self.model, 'c'):
            # get desired centroids and widths for z and y cavities; for standard single-cavity model these are the same
            # after the big fix to the eos call (feb 2021) these perform less well when sampling; the 'rstab' and 'rstab_in' scheme works better
            cz = self.model.c2 if hasattr(self.model, 'c2') else self.model.c
            cy = self.model.c1 if hasattr(self.model, 'c1') else self.model.c
            wz = self.model.w2 if hasattr(self.model, 'w2') else self.model.w
            wy = self.model.w1 if hasattr(self.model, 'w1') else self.model.w
        else:
            if hasattr(self.model, 'rstab_in'):
                # using alternate parameterization with stable region from r=rstab_in to r=rstab
                rstab_in = self.model.rstab_in
                rstab = self.model.rstab
                cz = cy = 0.5 * (self.model.rstab_in + self.model.rstab)
                wz = wy = self.model.rstab - self.model.rstab_in
            else:
                # using alternate parameterization with stable region from r=0 to r=rstab
                rstab = self.model.rstab
                cz = cy = 0.5 * rstab
                wz = wy = rstab

        if self.params['gradient_shape'] == 'sigmoid':
            self.model.z = self.model.z1 + (self.model.z2 - self.model.z1) * 0.5 * (1. + np.cos(np.pi * (cz - rf - wz / 2) / wz))
            self.model.z[rf < cz - wz / 2] = self.model.z2
            self.model.z[rf > cz + wz / 2] = self.model.z1
        elif self.params['gradient_shape'] == 'linear':
            self.model.z = self.model.z1 + (self.model.z2 - self.model.z1) * (0.5 + (cz - rf) / wz)
            self.model.z[rf < cz - wz / 2] = self.model.z2
            self.model.z[rf > cz + wz / 2] = self.model.z1
        else:
            raise ValueError(f"gradient_shape {self.params['gradient_shape']} not recognized")

        if self.model.y2_xy is None:
            y_xy = self.model.y1_xy * np.ones_like(rf)
        else:
            if self.params['gradient_shape'] == 'sigmoid':
                self.model.y_xy = self.model.y1_xy + (self.model.y2_xy - self.model.y1_xy) * 0.5 * (1. + np.cos(np.pi * (cy - rf - wy / 2) / wy))
                self.model.y_xy[rf < cy - wy / 2] = self.model.y2_xy
                self.model.y_xy[rf > cy + wy / 2] = self.model.y1_xy
            elif self.params['gradient_shape'] == 'linear':
                self.model.y_xy = self.model.y1_xy + (self.model.y2_xy - self.model.y1_xy) * (0.5 + (cy - rf) / wy)
                self.model.y_xy[rf < cy - wy / 2] = self.model.y2_xy
                self.model.y_xy[rf > cy + wy / 2] = self.model.y1_xy
            else:
                raise ValueError(f"gradient_shape {self.params['gradient_shape']} not recognized")

            y_xy = self.model.y1_xy + (self.model.y2_xy - self.model.y1_xy) * 0.5 * (1. + np.cos(np.pi * (cy - rf - wy / 2) / wy))
            y_xy[rf < cy - wy / 2] = self.model.y2_xy
            y_xy[rf > cy + wy / 2] = self.model.y1_xy

        self.model.y_xy[self.model.y_xy == 0.] = 1e-20 # keeps mixture eos evaluable
        self.model.y = self.model.y_xy * (1. - self.model.z)

        # import pickle
        # with open('debug.pkl', 'wb') as f:
        #     pickle.dump({'r':self.l, 'z':z, 'y_xy':y_xy}, f)

        assert np.all(self.model.y_xy > 0)
        assert np.all(self.model.y_xy < 1)

        if np.any(np.isnan(self.model.grada)):
            print('nans in grada: y1={} y2={} z1={} z2={}'.format(self.model.y1, self.model.y2, self.model.z1, self.model.z2))
            raise ValueError

        try:
            assert not np.any(np.isnan(self.p)), 'nans in p during adjust mcore, before t integration'
            assert not np.any(np.isnan(self.model.t)), 'nans in t during adjust mcore, before t integration'
            assert not np.any(np.isnan(self.model.grada)), 'nans in grada during adjust mcore, before t integration'
        except AssertionError as e:
            raise EOSError(e.args[0])

        # hhe_res = self.model.hhe_eos.get(np.log10(self.model.p), np.log10(self.model.t), y)
        hhe_res = self.model.hhe_eos.get(np.log10(self.model.p), np.log10(self.model.t), self.model.y_xy) # was previously y ;(
        z_res = self.model.z_eos.get(np.log10(self.model.p), np.log10(self.model.t), self.model.f_ice)

        # update densities
        rho_z = 10 ** z_res['logrho']
        rho_hhe = 10 ** hhe_res['logrho']
        rhoinv = self.model.z / rho_z + (1. - self.model.z) / rho_hhe
        self.rho[:] = rhoinv[:] ** -1.

        self.model.grada = hhe_res['grada']

        if 'superad' in self.params and self.params['superad']:
            gradyp = np.gradient(np.log(self.model.y_xy)) / np.gradient(np.log(self.p))
            gradz = np.gradient(np.log(self.model.z)) / np.gradient(np.log(self.p))
            # these are both positive by construction for our model; just look at sum to decide which regions are stable
            self.model.gradt = self.model.grada + self.params['superad'] * np.sign(gradyp + gradz)
        else:
            # un_grav repo gets grada this way:
            # rhot = self.rho / rho_z * z * z_res['rhot'] + self.rho / rho_hhe * (1. - z) * hhe_res['rhot']
            # delta = -rhot
            # cp = z * z_res['cp'] + (1. - z) * hhe_res['cp']
            # self.model.grada = self.p * delta / self.model.t / self.rho / cp

            self.model.gradt = self.model.grada

        # use linear interpolant for fast evaluations of gradt at arbitrary pressures.
        # allow it to extrapolate just because, for whatever reason, solve_ivp is passing pressures
        # barely larger (0.1 ppm) than the central pressure in rare cases.
        interp_gradt = interp1d(self.p, self.model.gradt, fill_value='extrapolate')
        def dtdp(p, t):
            # print(f'{p:e} {self.p[0]:e} {self.p[0]-p:e}')
            return t / p * interp_gradt(p)
        p_eval = self.p[::-1] # integrate from surface to center
        # sanity checks
        assert not np.any(np.isnan(self.p))
        assert not np.any(self.p[1:] > self.p[0])
        assert not np.any(self.p[:-1] < self.p[-1])
        assert self.p[-1] == 1e6
        assert interp_gradt(p_eval[0])
        assert interp_gradt(p_eval[-1])
        # integrate grada to get t profile
        sol = solve_ivp(dtdp, (p_eval[0], p_eval[-1]), np.array([self.model.t1]), t_eval=p_eval)
        assert sol.success, 'failed in integrate_temperature'
        self.model.t = sol.y[0][::-1]


        if np.any(np.isnan(self.rho)):
            raise ToFAdjustError('nans in rho during adjust; possibly off eos tables')
            print('logp where rho==nan', np.log10(self.p[np.isnan(self.rho)]))
            print('logt where rho==nan', np.log10(self.model.t[np.isnan(self.rho)]))
            print('chi_rho where rho==nan', chi_rho[np.isnan(self.rho)])
            print('chi_t where rho==nan', chi_t[np.isnan(self.rho)])
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

    def save_model(self):
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
        self.model.ymean_xy = self.model.mhe / (self.mtot_calc - self.model.mz) # current actual M_He / (M_H + M_He)
        self.model.ym = self.model.mhe / self.mtot_calc # current actual M_He / M_tot

        if False: # now our solutions include models with z_center < 0.5, so the old definition won't work
            # esimates of core mass and core radius, written with a smooth model in mind;
            # in the case of a three-layer model should reduce to the specified mcore
            scalars['mcore_proxy'] = self.m_calc[self.model.z < 0.5][0]
            scalars['rcore_proxy'] = self.l[self.model.z < 0.5][0]
            mz = cumtrapz(self.model.z, x=self.m_calc, initial=0.)
            try:
                scalars['mz_in'] = mz[self.model.z > 0.5][-1]
                scalars['mz_out'] = mz[-1] - scalars['mz_in']
                my = cumtrapz(self.model.y, x=self.m_calc, initial=0.)
                scalars['my_in'] = my[self.model.z > 0.5][-1]
                scalars['my_out'] = my[-1] - scalars['my_in']
                scalars['rhoc_proxy'] = self.rho[self.model.z < 0.5][0]
                scalars['pc_proxy'] = self.p[self.model.z < 0.5][0]
                scalars['tc_proxy'] = self.model.t[self.model.z < 0.5][0]
            except IndexError:
                # raise ToFAdjustError('no zone with Z>0.5')
                scalars['mz_in'] = 0.
                scalars['mz_out'] = 0.
                scalars['my_in'] = 0.
                scalars['my_out'] = 0.
                scalars['rhoc_proxy'] = 0.
                scalars['pc_proxy'] = 0.
                scalars['tc_proxy'] = 0.
        else:
            # define "core mass" and "core radius" as mass and radius of the boundary containing 50% of the model's total heavy element mass
            mz = cumtrapz(self.model.z, x=self.m_calc, initial=0.)
            mzf = mz / mz[-1]
            scalars['rcore_proxy'] = np.interp(0.5, mzf, self.l)
            scalars['mcore_proxy'] = np.interp(0.5, mzf, self.m_calc)
            scalars['mzcore_proxy'] = np.interp(0.5, mzf, mz)

        if hasattr(self.model, 'rstab'):
            scalars['mstab'] = np.interp(self.model.rstab, self.l / self.l[-1], self.m_calc)
            scalars['mzstab'] = np.interp(self.model.rstab, self.l / self.l[-1], mz)
        if hasattr(self.model, 'rstab_in'):
            scalars['mcore_proxy_inner'] = np.interp(self.model.rstab_in, self.l / self.l[-1], self.m_calc)
            scalars['mzcore_proxy_inner'] = np.interp(self.model.rstab_in, self.l / self.l[-1], mz)

        scalars['rho_center'] = self.rho[0]
        scalars['p_center'] = self.p[0]
        scalars['t_center'] = self.model.t[0]

        # scalar quantities that self.model may or may not have set, depending on model type:
        model_scalar_names = 'mcore', 'y1', 'y2', 'ym', 'kcore', 'k_shell_top', 'ktrans', \
            'z1', 'z2', 'mhe', 'mz', 'c', 'c2', 'y1_xy', 'y2_xy', 'max_y', 'r1', 'r2', 'r3'
        for name in model_scalar_names:
            if hasattr(self.model, name): scalars[name] = getattr(self.model, name)

        self.scalars = scalars # save as attribute so that MCMC can use for likelihood

        output = {}
        output['params'] = self.params
        output['scalars'] = self.scalars

        # add vector output
        vectors = {}
        vectors['l'] = l = self.l
        vectors['lf'] = self.l / self.l[-1]
        vectors['req'] = self.r_eq
        vectors['rpol'] = self.r_pol
        vectors['rho'] = rho = self.rho
        vectors['p'] = p = self.p
        # vectors['u'] = self.u # potential
        vectors['m_calc'] = m = self.m_calc
        # shape functions
        vectors['s0'] = self.s0
        vectors['s2'] = self.s2
        vectors['s4'] = self.s4
        vectors['s6'] = self.s6
        vectors['s8'] = self.s8
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
        y[y == 0.] = 1e-20
        yp = y / (1. - z) # Y^\prime = Y/(X+Y)=Y/(1-Z); this is referred to as y_xy elsewhere in the code

        # both forms of brunt calculation will require eos quantities
        hhe_eos = self.model.hhe_eos
        z_eos = self.model.z_eos
        hhe_res = hhe_eos.get(np.log10(p), np.log10(t), yp)
        z_res = z_eos.get(np.log10(p), np.log10(t), self.model.f_ice)

        vectors['gamma1'] = hhe_res['gamma1']
        grada = vectors['grada'] = self.model.grada # hhe_res['grada']
        vectors['rhot'] = hhe_res['rhot']
        vectors['g'] = g = m / l ** 2 * const.cgrav
        # vectors['geq'] = geq = m / self.r_eq ** 2 * const.cgrav
        # vectors['gpol'] = gpol = m / self.r_pol ** 2 * const.cgrav
        # vectors['entropy_hhe'] = 10 ** hhe_res['logs'] * const.mp / const.kb

        # vectors['dlnp_dr'] = np.gradient(np.log(p)) / np.gradient(l)
        # vectors['dlnrho_dr'] = np.gradient(np.log(rho)) / np.gradient(l)

        # vectors['n2_direct'] = g * (vectors['dlnp_dr'] / vectors['gamma1'] - vectors['dlnrho_dr'])
        # vectors['n2eq_direct'] = geq * (vectors['dlnp_dr'] / vectors['gamma1'] - vectors['dlnrho_dr'])
        # vectors['n2pol_direct'] = gpol * (vectors['dlnp_dr'] / vectors['gamma1'] - vectors['dlnrho_dr'])

        vectors['n2_direct'] = g ** 2 * rho / p * (np.gradient(np.log(rho)) / np.gradient(np.log(p)) - 1. / vectors['gamma1'])
        vectors['n2eq_direct'] = (self.l / self.r_eq) ** 4 * vectors['n2_direct'] # geq/g = (l/req) ** 2; second power of 2 comes from g**2 in n2
        vectors['n2pol_direct'] = (self.l / self.r_pol) ** 4 * vectors['n2_direct'] # gpol/g = (l/rpol) ** 2

        vectors['rho_h'] = rho_h = hhe_res['rho_h']
        vectors['rho_he'] = rho_he = hhe_res['rho_he']
        vectors['rho_hhe'] = rho_hhe = 10 ** hhe_res['logrho']
        vectors['rho_z'] = rho_z = 10 ** z_res['logrho']

        dlnrho_dlnz = rho * z * (1. / rho_hhe - 1. / rho_z) # constant p, t, yp
        dlnrho_dlnyp = rho * y * (1. / rho_h - 1. / rho_he) # constant p, t, z # equal to dlnrho_dlny since yp is proportional to y

        chi_rho_hhe = hhe_res['chirho']

        if True:
            # ignore the z part of chi_rho, chi_t - the compressibilities from z eos are a minor contribution and
            # numerics in our aneos implementation can give unphysical zero crossings, so better to leave out
            chi_rho = chi_rho_hhe
            chi_t = - hhe_res['rhot'] / hhe_res['rhop']
            dlnrho_dlnt_const_p = - hhe_res['rhot']
        else:
            chi_rho_z = z_res['chirho']
            chi_rho = (z * rho / rho_z / chi_rho_z + (1. - z) * rho / rho_hhe / chi_rho_hhe) ** -1.

            dlnrho_dlnt_const_p = z * rho / rho_z * z_res['rhot'] + (1. - z) * rho / rho_he * hhe_res['rhot']
            dlnrho_dlnp_const_t = z * rho / rho_z * z_res['rhop'] + (1. - z) * rho / rho_he * hhe_res['rhop']
            chi_t = - dlnrho_dlnt_const_p / dlnrho_dlnp_const_t

        if self.params['model_type'] == 'continuous':
            vectors['gradyp'] = gradyp = np.gradient(np.log(yp)) / np.gradient(np.log(p))
            vectors['gradz'] = gradz = np.gradient(np.log(z)) / np.gradient(np.log(p))
            # vectors['brunt_b'] = b = chi_rho / chi_t * (dlnrho_dlny * grady + dlnrho_dlnz * gradz)

            vectors['dlnrho_dlnyp'] = dlnrho_dlnyp
            vectors['dlnrho_dlnz'] = dlnrho_dlnz
            vectors['chi_t'] = chi_t
            vectors['chi_rho'] = chi_rho
            vectors['chi_rho_hhe'] = chi_rho_hhe
            vectors['delta'] = - dlnrho_dlnt_const_p # do save this one if going to save gyre info

            vectors['n2_yp'] = g ** 2 * rho / p * dlnrho_dlnyp * gradyp
            vectors['n2_z'] = g ** 2 * rho / p * dlnrho_dlnz * gradz

            vectors['n2_thermal'] = g ** 2 * rho / p * chi_t / chi_rho * (grada - gradt)
            vectors['n2_composition'] = vectors['n2_z'] + vectors['n2_yp']
            vectors['n2'] = vectors['n2_thermal'] + vectors['n2_composition']

            # equivalently we could use Y, Z as the composition variables and get the same answer for physical quantities like N^2, but the
            # relative contributions from Z and Y are different than they would be for Z and Y^\prime. in other words:
            vectors['grady'] = grady = np.gradient(np.log(y)) / np.gradient(np.log(p))
            vectors['n2_y_yz_basis'] = g ** 2 * rho / p * rho * y * (1. / rho_h - 1. / rho_he) * grady # different from vectors['n2_yp'] above because grad_Y != grad_Y'
            vectors['n2_z_yz_basis'] = g ** 2 * rho / p * rho * z * (1. / rho_h - 1. / rho_z) * gradz # different from vectors['n2_z'] above because (dlnrho/dlnp)_PTY != (dlnrho/dlnp)_PTY'

            vectors['n2eq'] = (self.l / self.r_eq) ** 4 * vectors['n2']
            vectors['n2pol'] = (self.l / self.r_pol) ** 4 * vectors['n2']

            n2 = np.copy(vectors['n2']) # copy because it may be abused below (e.g., zero out n2<0)
            n2[n2 < 0] = 0.
            w0 = np.sqrt(const.cgrav * scalars['mtot_calc'] / scalars['rm'] ** 3)
            scalars['max_n'] = np.sqrt(np.max(n2)) / w0
            scalars['max_neq'] = np.sqrt(np.max(vectors['n2eq'])) / w0
            scalars['max_npol'] = np.sqrt(np.max(vectors['n2pol'])) / w0

            scalars['pi0'] = np.pi ** 2 * 2 / trapz(np.sqrt(n2), x=np.log(l))
            scalars['mean_n2_gmode_cavity'] = trapz(l * n2) / trapz(l)

        elif self.params['model_type'] == 'three_layer': # special handling for all the zones with y or z or both equal to zero
            raise NotImplementedError('revisit calculation of brunt for three layer model before using.')
            y[y == 0] = 1e-50
            z[z == 0] = 1e-50
            grady = np.diff(np.log(y), append=0) / np.diff(np.log(p), append=0)
            gradz = np.diff(np.log(y), append=0) / np.diff(np.log(p), append=0)
            vectors['brunt_b'] = b = chi_rho / chi_t * (dlnrho_dlny * grady + dlnrho_dlnz * gradz)
            vectors['n2'] = g ** 2 * rho / p * chi_t / chi_rho * (grada - gradt + b)

            vectors['chi_t'] = chi_t
            vectors['chi_rho'] = chi_rho
            vectors['delta'] = - dlnrho_dlnt_const_p # do save this one if going to save gyre info
        else:
            raise ValueError('unhandled model_type in save_model.')

        scalars['ymean_xy'] = self.model.ymean_xy
        scalars['max_n'] = np.max(vectors['n2']) ** 0.5 / scalars['omega0']

        output['vectors'] = self.vectors = vectors
        output['scalars'] = self.scalars = scalars
        outfile = '{}/tof4_data.pkl'.format(self.output_path)
        if os.path.exists(outfile):
            raise ValueError('{} unexpectedly already exists'.format(outfile))
        with open(outfile, 'wb') as f:
            pickle.dump(output, f)
        if self.verbosity > 0:
            print('wrote {}'.format(outfile))
            # print(f'uid was {self.uid}')

    def write_gyre_model(self, downsample=False, outdir=None, rotation=False, brunt_option='default'):
        if not hasattr(self, 'scalars'):
            self.save_model() # in our usual pattern this would already have been called, though
            assert hasattr(self, 'vectors'), 'expected vectors to be set following save_model call'
        if outdir is None: outdir = f'output/{self.uid}'

        vec = self.vectors.copy()

        n = downsample if downsample else len(vec['l'])
        mtot = self.scalars['mtot_calc']
        rtot = self.scalars['rm']
        ltot = 1.
        version = 101 # see format spec in gyre-5.2/doc/mesa-format.pdf

        # ['l', 'req', 'rpol', 'rho', 'p', 'm_calc', 'y', 'z', 't', 'gamma1', 'grada', 'g', 'dlnp_dr', 'dlnrho_dr', 'gradt', 'n2']
        vec['k'] = np.arange(n)
        vec['lum'] = np.ones(n) # adiabatic mode calculation doesn't need luminosity

        if not rotation:
            # leave rotation out of gyre calculation; will apply corrections after the fact
            vec['omega'] = np.zeros(n)
        else:
            # rigid rotation
            vec['omega'] = np.ones(n) * (self.small * const.cgrav * mtot / rtot ** 3) ** 0.5
            print(vec['omega'][0])
        outfile = f'{outdir}/model.gyre'

        if False and np.any(np.diff(np.log(vec['rho'])) < -0.1): # jumps exist, record discontinuous quantities on either side
            jumps = {}
            ki = np.where(np.diff(np.log(vec['rho'])) < -0.1)[0]
            li = vec['l'][ki]
            for i, kval in enumerate(ki):
                jumps[i] = {}
                for qty in 'rho', 'gradt', 'gamma1', 'grada', 'delta':
                    jumps[i][qty] = vec[qty][kval], vec[qty][kval+1]
        else:
            ki = []
            li = []
            jumps = []

        # splines = {}
        # for qty in 'm_calc', 'p', 't', 'n2': # store cubic splines before we mess with anything
        #     splines[qty] = splrep(vec['l'], vec[qty], k=3)

        # if downsampling, must resample all quantities other than k, lum, omega
        new_l = np.linspace(vec['l'][0], vec['l'][-1], n)
        # new_l = np.linspace(1e-4 * vec['l'][-1], n)
        brunt_key = {
            'default':'n2',
            'equatorial':'n2eq',
            'direct':'n2_direct'
        }[brunt_option]
        for qty in 'm_calc', 'p', 't', 'rho', 'gradt', brunt_key, 'gamma1', 'grada', 'delta':
            vec[qty] = splev(new_l, splrep(vec['l'], vec[qty], k=1)) # was k=3 previously, changed 02222021
        vec['l'] = new_l

        if np.any(ki):
            for i, lval in enumerate(li):
                k = np.where(vec['l'] < lval)[0][-1] + 1
                # add two mesh points both at l==li
                for qty in 'm_calc', 'p', 't', brunt_key, 'lum', 'omega':
                    # vec[qty] = np.insert(vec[qty], k, splev(lval, splines[qty]))
                    vec[qty] = np.insert(vec[qty], k, vec[qty][k])
                    vec[qty] = np.insert(vec[qty], k, vec[qty][k])
                for qty in list(jumps[i]):
                    # print(i, qty, jumps[i][qty])
                    lo, hi = jumps[i][qty]
                    vec[qty] = np.insert(vec[qty], k, hi)
                    vec[qty] = np.insert(vec[qty], k, lo)
                vec['l'] = np.insert(vec['l'], k, lval)
                vec['l'] = np.insert(vec['l'], k, lval)
                vec['k'] = np.insert(vec['k'], k+1, k+2)
                vec['k'] = np.insert(vec['k'], k+1, k+1)
                vec['k'][k+3:] += 2

                vec['rho'][k+2] = 0.5 * (vec['rho'][k+1] + vec['rho'][k+3]) # splev(vec['l'][k+2], splrep(np.delete(vec['l'], k+2), np.delete(vec['rho'], k+2), k=3))
                vec['p'][k+2] = 0.5 * (vec['p'][k+1] + vec['p'][k+3]) # splev(vec['l'][k+2], splrep(np.delete(vec['l'], k+2), np.delete(vec['rho'], k+2), k=3))

        n = len(vec['k'])
        # now set header. n may have been updated if density discontinuities are present.
        header = '{:>5n} {:>16.8e} {:>16.8e} {:>16.8e} {:>5n}\n'.format(n, mtot, rtot, ltot, version)

        with open(outfile, 'w') as fw:
            fw.write(header)

            ncols = 19
            for k in vec['k']:
                data_fmt = '{:>5n} ' + '{:>16.8e} ' * (ncols - 1) + '\n'
                # debugging the three layer case (erroneous arithmetic operation in gyre)
                # if vec['gradt'][k] < 0: vec['gradt'][k] = 0
                # if vec['grada'][k] < 0: vec['grada'][k] = 0
                # if abs(vec['gradt'][k]) < 1e-6: vec['gradt'][k] = 0
                # if abs(vec['grada'][k]) < 1e-6: vec['grada'][k] = 0

                if abs(vec[brunt_key][k]) < 1e-12: vec[brunt_key][k] = 1e-12 # 07202020: previously had 0, which can give gyre a hard time # reduce to 1e-20 02222021
                if vec[brunt_key][k] < 0.: vec[brunt_key][k] = 1e-12 # 08252020 # 02222021

                if self.params['model_type'] == 'three_layer':
                    # print('beep')
                    vec[brunt_key][k] = 0. # let reconstruct_As in gyre handle this
                    if vec['l'][k] < li[0]:
                        vec['gradt'][k] = 0
                        vec['grada'][k] = 0.35
                    if vec['grada'][k] <= 0.:
                        vec['grada'][k] = 0.35

                data = k+1, vec['l'][k], vec['m_calc'][k], vec['lum'][k], vec['p'][k], vec['t'][k], vec['rho'][k], \
                    vec['gradt'][k], vec[brunt_key][k], vec['gamma1'][k], vec['grada'][k], vec['delta'][k], \
                    1, 0, 0, 1, 0, 0, \
                    vec['omega'][k]
                fw.write(data_fmt.format(*data))

    def write_pseudo_model(self, prefix='sat', downsample=False, outdir=None, brunt_key='n2'):
        if not hasattr(self, 'scalars'):
            self.save_model() # in our usual pattern this would already have been called, though
            assert hasattr(self, 'vectors'), 'expected vectors to be set following save_model call'
        if outdir is None: outdir = f'output/{self.uid}'

        f = (self.r_eq - self.r_pol) / self.r_eq # (v['req']-v['rpol']) / v['req'] # the flattening "f", about 10% for Saturn.
        # when Dahlen & Tromp 1998 and Vorontsov & Zharkov 1981 present the Radau-Darwin approximation
        # to the Clairaut equation they call this the ellipticity.
        # the definition of ellipticity in Fuller 2014 is different by a factor of 2/3.
        # ell(f) follows from the expansions of a and b in polynomials and figure functions s_2n, neglecting terms of order s_2^2, s_4, etc.:
        tof_ell = -1. / (0.5 - 1.5 / f) # or about 2/3 * f
        tof_eta = np.gradient(tof_ell) / np.gradient(self.l) * self.l / tof_ell

        mtot = self.scalars['mtot_calc']
        rtot = self.scalars['rm']
        small = self.scalars['small']
        # max_n = np.max(self.vectors[brunt_key] ** 0.5)
        with open(f'{outdir}/{prefix}.scalars', 'w') as fw:
            fw.write(f'{mtot:12.6e} {rtot:12.6e} {small:8.5f} {brunt_key}\n')

        outfile = f'{outdir}/{prefix}.in'

        columns = {}
        if downsample:
            new_l = np.linspace(self.l[0], self.l[-1], downsample)
            columns['r'] = new_l
            columns['m_calc'] = np.interp(new_l, self.l, self.m_calc)
            columns['p'] = np.interp(new_l, self.l, self.p)
            columns['rho'] = np.interp(new_l, self.l, self.rho)
            columns['n2'] = np.interp(new_l, self.l, self.vectors[brunt_key])
            columns['gamma1'] = np.interp(new_l, self.l, self.vectors['gamma1'])
            columns['ell'] = np.interp(new_l, self.l, tof_ell)
            columns['eta'] = np.interp(new_l, self.l, tof_eta)
        else:
            columns['r'] = np.copy(self.l)
            columns['m_calc'] = np.copy(self.m_calc)
            columns['p'] = np.copy(self.p)
            columns['rho'] = np.copy(self.rho)
            columns['n2'] = np.copy(self.vectors[brunt_key])
            columns['gamma1'] = np.copy(self.vectors['gamma1'])
            columns['ell'] = tof_ell
            columns['eta'] = tof_eta
        with open(outfile, 'w') as fw:
            for k, _ in enumerate(columns['r']):
                for key in list(columns):
                    fw.write(f"{columns[key][k]:16e} ")
                fw.write('\n')

    def set_req_rpol(self):
        '''
        calculate equatorial and polar radius vectors from the figure functions s_2n and legendre polynomials P_2n.
        see N17 eq. (B.1) or ZT78 eq. (27.1).
        also calculates q from m and new r_eq[-1].
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

        if self.method_for_aa2n_solve == 'full' or force_full:

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
            fskip = int(self.method_for_aa2n_solve.split()[1])
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

            # cubic interpolants for approximate shape functions
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
        # for comparison calculate the first-order ellipticity from Clairaut theory w/ the
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

    '''test a single model'''

    omega0 = np.sqrt(const.cgrav * const.msat / const.rsat ** 3)
    omega_sat = np.pi * 2 / (10.561 * 3600) # 10h 33m 38s = 10.56055 h
    small = (omega_sat / omega0) ** 2 # sometimes denoted m; m_rot in the paper

    z1, rstab, y2_xy, f_ice = 0.05927145799416273,    0.7068565323794515,    0.855706683903526,    0.04392185816793537    # from best model in chain n4

    params = {}
    params['small'] = small
    params['adjust_small'] = True # tells the model to adjust the nondimensional spin parameter to preserve *dimensional* spin frequency as the model's mean radius changes during iterations
    params['mtot'] = const.msat
    params['req'] = 60268e5
    params['nz'] = 4096
    params['verbosity'] = 1
    params['t1'] = 135.
    params['f_ice'] = f_ice

    # composition choices
    if False: # make a three-layer homogeneous model: core, inner envelope, outer envelope
        params['model_type'] = 'three_layer'
        params['y1'] = 0.12 # outer envelope helium mass fraction
        params['ymean_xy'] = 0.275 # M_He / (M_H + M_He); outer iterations will adjust y2 to satisfy this
        params['z1'] = 0.5 # outer envelope heavy element mass fraction
        params['z2'] = 0.9 # inner envelope heavy element mass fraction
        params['mcore'] = 10. # initial guess for core mass (earth masses); will be adjusted in iterations to satisfy correct total mass
        params['pt'] = 0.1 # pressure (Mbar) corresponding to inner/envelope envelope boundary
    elif False: # make a model with a single stable region over which Z and Y':=Y/(X+Y) both vary
        params['model_type'] = 'continuous'
        params['gradient_shape'] = 'sigmoid'
        params['y2_xy'] = 0.95 # 0.275+1e-6 # y/(x+y) in inner core
        params['ymean_xy'] = 0.275 # ymean_xy # M_He / (M_H + M_He); outer iterations will adjust y1_xy to satisfy this
        params['y1_xy'] = 0.27 # initial guess for y/(x+y) in outer envelope; will be adjusted in iterations to satisfy specified ymean_xy
        params['z1'] = z1 # outer envelope heavy element mass fraction
        params['z2'] = z2 # inner core heavy element mass fraction
        params['c'] = 0.4 # initial guess for centroid radius of stable region; will be adjusted in iterations to satisfy correct total mass
        params['w'] = w # radial width of stable region
    elif True: # same, but inner cavity goes from r=0 to r=rstab; z2 will be adjusted to satisfy total mass
        params['model_type'] = 'continuous'
        params['gradient_shape'] = 'sigmoid'
        params['y2_xy'] = y2_xy # y/(x+y) in inner core
        params['ymean_xy'] = 0.275 # ymean_xy # M_He / (M_H + M_He); outer iterations will adjust y1_xy to satisfy this
        params['y1_xy'] = 0.27 # initial guess for y/(x+y) in outer envelope; will be adjusted in iterations to satisfy specified ymean_xy
        params['z1'] = z1 # outer envelope heavy element mass fraction # starting guess, will vary to satisfy total mass
        params['z2'] = 0.5 # inner core heavy element mass fraction
        # params['c'] = 0.4 # don't set these for this model type
        # params['w'] = 0.6 # don't set these for this model type
        params['rstab'] = rstab
        # params['rstab_in'] = rstab_in
    else: # make a model with an inner cavity over which Z varies and and outer cavity over which Y' varies
        params['model_type'] = 'continuous'
        params['gradient_shape'] = 'sigmoid'
        params['y2_xy'] = y2_xy # y/(x+y) in inner core
        params['ymean_xy'] = ymean_xy # M_He / (M_H + M_He); outer iterations will adjust y1_xy to satisfy this
        params['y1_xy'] = 0.27 # initial guess for y/(x+y) outside outer cavity; will be adjusted in iterations to satisfy specified ymean_xy
        params['z1'] = z1 # heavy element mass fraction outside inner cavity
        params['z2'] = z2 # heavy element mass fraction within inner cavity
        params['c2'] = 0.4 # initial guess for centroid radius of inner stable region; will be adjusted in iterations to satisfy correct total mass
        params['w2'] = w # radial width of inner stable region
        params['c1'] = 0.8 # centroid radius of outer stable region
        params['w1'] = 0.05 # radial width of outer stable region

    # relative tolerances
    params['j2n_rtol'] = 1e-5
    params['ymean_rtol'] = 1e-4
    params['mtot_rtol'] = 1e-5

    params['use_gauss_lobatto'] = True

    # initialize eos objects once, can pass to many tof4 objects
    try:
        import mh13_scvh
        hhe_eos = mh13_scvh.eos()
        import aneos_mix
        z_eos = aneos_mix.eos()
    except OSError:
        raise Exception('failed to initialize eos; did you unpack eos_data.tar.gz?')

    # finally make a tof4 instance and relax the model
    t = tof4(hhe_eos, z_eos, params)
    t.initialize_model()
    t.initialize_tof_vectors()
    t.relax()
    # t.write_gyre_model() # if you want to write a model in gyre format
