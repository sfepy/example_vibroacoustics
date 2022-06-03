# Vibroacoustics
#
# E.Rohan, V.Lukeš
# Homogenization of the vibro–acoustic transmission on periodically
# perforated elastic plates with arrays of resonators.
# https://arxiv.org/abs/2104.01367 (arXiv:2104.01367v1)
# https://doi.org/10.1016/j.apm.2022.05.040 (Applied Mathematical Modelling, 2022)
#
# compatible with SfePy 2022.1

import os.path as op
import numpy as nm
from acoustics_macro_utils import get_homogmat
from sfepy.homogenization.utils import define_box_regions
from sfepy.discrete.fem import Mesh
from sfepy.discrete.fem.periodic import match_y_plane, match_x_plane

wdir = op.dirname(__file__)


def get_regions(filename_mesh):
    mesh = Mesh.from_file(filename_mesh)
    bbox = nm.array(mesh.get_bounding_box())
    region_lb, region_rt = bbox

    return define_box_regions(2, region_lb, region_rt)


def get_homogmat_plate(coors, mode, pb):
    if mode != 'qp':
        return

    mconf = pb.conf.mconf
    c = mconf.sound_speed
    wave_num = mconf.wave_num
    rho0 = mconf.rho0

    c2 = c**2
    w = wave_num * c
    w2 = w**2

    pb.ofn_trunk = mconf.ofn_trunk + '_plate'

    out_ac = get_homogmat(coors, mode, pb, mconf.coefs_filename, omega=w)

    nqp = coors.shape[0]
    out = {}

    out['A'] = out_ac['A']
    out['w2F'] = out_ac['F'] * w2
    out['wB'] = out_ac['B'] * w

    vol_Imp = 0.5 * (out_ac['Vol_Imp']['volume_Im'] +
                     out_ac['Vol_Imp']['volume_Ip'])
    zeta = out_ac['volumes']['volume_Y'] / vol_Imp

    out['w2Kr'] = nm.tile(zeta / c2, (nqp, 1, 1)) * w2
    out['w'] = nm.ones((nqp, 1, 1), dtype=nm.float64) * w

    out_vc = get_homogmat(coors, mode, pb, mconf.coefs_filename_plate)

    bar_h = out_ac['h']
    h = mconf.eps0 * bar_h

    out['E'] = bar_h / rho0 * out_vc['Cm']
    out['h2E'] = h**2 / 12. * out['E']
    out['wH'] = bar_h * out_vc['Hm'] * w
    out['w2K'] = out['w2Kr'] + rho0 * out_vc['Km'] * bar_h * w2
    out['S'] = bar_h / rho0 * out_vc['Gm']
    sfdim = out_vc['Gm'].shape[2]

    out['w2C3'] = out_ac['C'][:, sfdim:, :] * w2
    out['wD'] = nm.ascontiguousarray(out_ac['D'][:, :sfdim, :sfdim]) * w
    out['w2M'] = (out_ac['tM'] + out_ac['M'][:, :sfdim, :sfdim]) * w2
    out['w2N'] = out_ac['M'][:, sfdim:, sfdim:] * w2
    out['w2h2L'] = h**2 / 12. * out_ac['tM'] * w2

    print('### material-plate: wave number = ', wave_num)

    return out


def define(**kwargs):
    mconf = kwargs['master_problem'].conf

    filename_mesh = mconf.filename_mesh_plate

    regions = {
        'Gamma0_1': 'all',
    }

    regions.update(get_regions(filename_mesh))

    functions = {
        'get_homogmat':
            (lambda ts, coors, mode=None, problem=None, **kwargs:
             get_homogmat_plate(coors, mode, problem),),
        'match_y_plane': (match_y_plane,),
    }

    materials = {
        'ac': 'get_homogmat',
    }

    fields = {
        'tvelocity0': ('complex', 'scalar', 'Gamma0_1', 1),
        'pressure0': ('complex', 'scalar', 'Gamma0_1', 1),
        'deflection': ('complex', 'scalar', 'Gamma0_1', 2),
        'displacement': ('complex', 'vector', 'Gamma0_1', 1),
        'rotation': ('complex', 'vector', 'Gamma0_1', 1),
    }

    integrals = {
        'i': 4,
    }

    variables = {
        'sp0': ('unknown field', 'pressure0', 0),
        'sq0': ('test field', 'pressure0', 'sp0'),
        'dp0': ('unknown field', 'pressure0', 1),
        'dq0': ('test field', 'pressure0', 'dp0'),
        'g01': ('unknown field', 'tvelocity0', 2),
        'f01': ('test field', 'tvelocity0', 'g01'),
        'g02': ('unknown field', 'tvelocity0', 3),
        'f02': ('test field', 'tvelocity0', 'g02'),
        'u': ('unknown field', 'displacement', 4),
        'v': ('test field', 'displacement', 'u'),
        'w': ('unknown field', 'deflection', 5),
        'z': ('test field', 'deflection', 'w'),
        'theta': ('unknown field', 'rotation', 6),
        'nu': ('test field', 'rotation', 'theta'),
    }

    ebcs = {
        'fixed_l': ('Left', {'w.0': 0.0, 'u.all': 0.0, 'theta.all': 0.0}),
        'fixed_r': ('Right', {'w.0': 0.0, 'u.all': 0.0, 'theta.all': 0.0}),
    }

    epbcs = {
        # 'per_g01': (['Bottom', 'Top'], {'g01.0': 'g01.0'},
        #             'match_y_plane'),
        # 'per_g02': (['Bottom', 'Top'], {'g02.0': 'g02.0'},
        #             'match_y_plane'),
        # 'per_dp0': (['Bottom', 'Top'], {'dp0.0': 'dp0.0'},
        #             'match_y_plane'),
        # 'per_sp0': (['Bottom', 'Top'], {'sp0.0': 'sp0.0'},
        #             'match_y_plane'),
        'per_w': (['Bottom', 'Top'], {'w.0': 'w.0'},
                  'match_y_plane'),
        'per_u': (['Bottom', 'Top'], {'u.all': 'u.all'},
                  'match_y_plane'),
        'per_theta': (['Bottom', 'Top'], {'theta.all': 'theta.all'},
                      'match_y_plane'),
    }

    equations = {
        # p^0 = 0.5 * (P^+ + P^-)
        # eq. (79)_1
        'eq_g01': """
                0.5 * dw_diffusion.i.Gamma0_1(ac.A, f01, sp0)
            - 0.5 * dw_dot.i.Gamma0_1(ac.w2K, f01, sp0)
            + %s * dw_v_dot_grad_s.i.Gamma0_1(ac.wD, u, f01)
            + %s * dw_biot.i.Gamma0_1(ac.wH, u, f01)
            + %s * dw_dot.i.Gamma0_1(ac.w, f01, g01)
            - %s * dw_dot.i.Gamma0_1(ac.w, f01, g02)
                    = 0""" % (1j, 1j, 1j / mconf.eps0, 1j / mconf.eps0),
        # eq. (80)_1
        'eq_g02': """
            + 0.5 * dw_dot.i.Gamma0_1(ac.w2F, f02, g01)
            + 0.5 * dw_dot.i.Gamma0_1(ac.w2F, f02, g02)
                    - dw_dot.i.Gamma0_1(ac.w2C3, f02, w)
            - %s * dw_dot.i.Gamma0_1(ac.w, f02, dp0)
                    = 0""" % (1j / mconf.eps0,),
        # p^0 = 0.5 * (P^+ + P^-)
        # eq. (79)_2
        'eq_v': """
            - %s * dw_v_dot_grad_s.i.Gamma0_1(ac.wD, v, sp0)
            - %s * dw_biot.i.Gamma0_1(ac.wH, v, sp0)
                    + dw_lin_elastic.i.Gamma0_1(ac.E, v, u)
                    - dw_dot.i.Gamma0_1(ac.w2M, v, u)
                    = 0""" % (1j * 0.5, 1j * 0.5),
        # eq. (80)_2
        'eq_z': """
                    - dw_dot.i.Gamma0_1(ac.w2N, z, w)
                    + dw_diffusion.i.Gamma0_1(ac.S, z, w)
                    - dw_v_dot_grad_s.i.Gamma0_1(ac.S, theta, z)
            + 0.5 * dw_dot.i.Gamma0_1(ac.w2C3, z, g01)
            + 0.5 * dw_dot.i.Gamma0_1(ac.w2C3, z, g02)
                    = 0""",
        # eq. (80)_2
        'eq_nu': """
                    - dw_dot.i.Gamma0_1(ac.w2h2L, nu, theta)
                    + dw_lin_elastic.i.Gamma0_1(ac.h2E, nu, theta)
                    + dw_dot.i.Gamma0_1(ac.S, nu, theta)
                    - dw_v_dot_grad_s.i.Gamma0_1(ac.S, nu, w)
                    = 0""",
    }

    options = {
        'output_dir': mconf.options['output_dir'],
    }

    solvers = {
        'ls': ('ls.scipy_direct', {}),
        'newton': ('nls.newton', {'i_max': 1,
                                  'eps_a': 1e-6,
                                  'eps_r': 1e-6,
                                  'problem': 'nonlinear', })
    }

    return locals()
