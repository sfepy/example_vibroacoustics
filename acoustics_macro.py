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
from collections.abc import Iterable
from scipy.io import savemat, loadmat
from sfepy.base.base import output, debug, Struct
from sfepy import data_dir
from sfepy.discrete.fem.periodic import match_y_plane, match_x_plane
from acoustics_macro_utils import eval_phi, post_process,\
    generate_plate_mesh, get_region_entities
from sfepy.discrete.projections import project_by_component
from sfepy.discrete.fem import Mesh, FEDomain

wdir = op.dirname(__file__)


def post_process_macro(out, pb, state, extend=False):
    pbvars = pb.get_variables()

    n1, ng1, c1, cg1, ds1, nmap1 = get_region_entities(pbvars['p1'])
    noff = n1.shape[0]
    n2, ng2, c2, cg2, _, nmap2 = get_region_entities(pbvars['p2'], noff=noff)
    nend = nm.max(c2) + 1
    nmap = nm.hstack([nmap1, nmap2])
    n1[:, 2] += pb.conf.eps0 * 0.5
    n2[:, 2] -= pb.conf.eps0 * 0.5

    mesh2 = Mesh.from_data('m2', nm.vstack([n1, n2]), nm.hstack([ng1, ng2]),
                           [nm.vstack([c1, c2])], [nm.hstack([cg1, cg2])],
                           [ds1])

    oname = op.join(pb.output_dir, pb.ofn_trunk + '_p.vtk')
    out2 = {}

    for ir in ['real.', 'imag.']:
        pdata = nm.zeros((nmap.shape[0], 1), dtype=nm.float64)
        for v, idxs in [('p1', slice(0, noff)), ('p2', slice(noff, nend))]:
            pdata[idxs, :] = out[ir + v].data

        out2[ir + 'p'] = Struct(name='p', mode='vertex', data=pdata)

    mesh2.write(oname, out=out2)

    post_process(out, pb, state, save_var0='dp0')

    for k1 in ['g01', 'imag.g01', 'real.g01']:
        o = out[k1]
        k0 = k1.replace('01', '0')
        k2 = k1.replace('01', '02')
        out[k0] = Struct(name=o.name,
                         mode=o.mode,
                         var_name=o.var_name,
                         data=(out[k1].data - out[k2].data) / pb.conf.eps0)

    for k in ['', 'imag.', 'real.']:
        o = out[k + 'dp0']
        k0 = k + 'jP1'
        out[k0] = Struct(name=o.name,
                         mode=o.mode,
                         var_name=o.var_name,
                         data=o.data / pb.conf.eps0)

        o = out[k + 'g01']
        o2 = out[k + 'g02']
        out[k + 'G1'] = Struct(name=o.name,
                               mode=o.mode,
                               var_name=o.var_name,
                               data=(o.data - o2.data) / pb.conf.eps0)

    return out


def get_mat(coors, mode, pb):
    if mode == 'qp':
        conf = pb.conf
        c = conf.sound_speed
        w = conf.wave_num * c
        nqp = coors.shape[0]
        aux = nm.ones((nqp, 1, 1), dtype=nm.float64)

        out = {
            'c2': aux * c**2,
            'w2': aux * w**2,
            'wc': aux * w * c,
            'wc2': aux * w * c**2,
        }

        print('### material: wave number = ', conf.wave_num)

        return out


def param_w(pb):
    out = []
    tl_out = []
    conf = pb.conf
    ofn_trunk = pb.ofn_trunk

    for k in conf.wave_nums:
        print('### wave number: ', k)
        conf.wave_num = k

        pb.ofn_trunk = ofn_trunk + '_w%d' % (k * pb.conf.sound_speed)
        pb.conf.ofn_trunk = pb.ofn_trunk
        yield pb, out

        state = out[-1][1].get_state_parts()
        tl_out.append(eval_phi(pb, state['p1'], state['p2'], conf.p_inc))
        print('>>> TL: ', tl_out[-1])

        yield None

    savemat(op.join(wdir, 'results', 'tloss.mat'),
            {'k': conf.wave_nums, 'tl': tl_out})


############################################################
def define(filename_mesh=None, sound_speed=None, rho0=None,
           freqs=None, p_inc=None, eps0=None,
           coefs_filename=None, coefs_filename_plate=None):

    # generate mid mesh
    filename_mesh_plate = generate_plate_mesh(op.join(wdir, filename_mesh))

    wave_num = nm.array(freqs) / sound_speed
    wave_nums, wave_num = wave_num, wave_num[0]

    regions = {
        'Omega1': 'cells of group 1',
        'Omega2': 'cells of group 2',
        'GammaIn': ('vertices of group 1', 'facet'),
        'GammaOut': ('vertices of group 2', 'facet'),
        'Gamma_aux': ('r.Omega1 *v r.Omega2', 'facet'),
        'Gamma0_1': ('copy r.Gamma_aux', 'facet', 'Omega1'),
        'Gamma0_2': ('copy r.Gamma_aux', 'facet', 'Omega2'),
        'Recovery': ('copy r.Gamma0_1', 'facet'),
        }

    fields = {
        'pressure1': ('complex', 'scalar', 'Omega1', 1),
        'pressure2': ('complex', 'scalar', 'Omega2', 1),
        'tvelocity0': ('complex', 'scalar', 'Gamma0_1', 1),
        'pressure0': ('complex', 'scalar', 'Gamma0_1', 1),
        'vfield1': ('complex', 'vector', 'Omega1', 1),
        'vfield2': ('complex', 'vector', 'Omega2', 1),
    }

    variables = {
        'p1': ('unknown field', 'pressure1', 0),
        'q1': ('test field', 'pressure1', 'p1'),
        'p2': ('unknown field', 'pressure2', 1),
        'q2': ('test field', 'pressure2', 'p2'),
        'sp0': ('unknown field', 'pressure0', 2),
        'sq0': ('test field', 'pressure0', 'sp0'),
        'dp0': ('unknown field', 'pressure0', 3),
        'dq0': ('test field', 'pressure0', 'dp0'),
        'g01': ('unknown field', 'tvelocity0', 4),
        'f01': ('test field', 'tvelocity0', 'g01'),
        'g02': ('unknown field', 'tvelocity0', 5),
        'f02': ('test field', 'tvelocity0', 'g02'),
        'P1': ('parameter field', 'pressure1', '(set-to-None)'),
        'P2': ('parameter field', 'pressure2', '(set-to-None)'),
        's1': ('parameter field', 'pressure1', '(set-to-None)'),
        's2': ('parameter field', 'pressure2', '(set-to-None)'),
        'v1': ('parameter field', 'vfield1', '(set-to-None)'),
        'v2': ('parameter field', 'vfield2', '(set-to-None)'),
    }

    integrals = {
        'i': 2,
    }

    ebcs = {}

    functions = {
        'get_mat': (lambda ts, coors, mode=None, problem=None, **kwargs:
                    get_mat(coors, mode, problem),),
        'match_y_plane': (match_y_plane,),
    }

    materials = {
        'ac': 'get_mat',
    }

    regions.update({
        'Near': ('vertices of group 3', 'facet'),
        'Far': ('vertices of group 4', 'facet'),
    })
    epbcs = {
        'per_p1': (['Near', 'Far'], {'p1.0': 'p1.0'}, 'match_y_plane'),
        'per_p2': (['Near', 'Far'], {'p2.0': 'p2.0'}, 'match_y_plane'),
    }

    options = {
        'output_dir': op.join(wdir, 'results'),
        'file_per_var': True,
        'post_process_hook': 'post_process_macro',
        'parametric_hook': 'param_w',
    }

    # p1 = P^+, p2 = P^-
    equations = {
        'eq_p1': """
              dw_laplace.i.Omega1(ac.c2, q1, p1)
            - dw_dot.i.Omega1(ac.w2, q1, p1)
       + %s * dw_dot.i.GammaOut(ac.wc, q1, p1)
       - %s * dw_dot.i.Gamma0_1(ac.wc2, q1, g01)
            = 0""" % (1j, 1j),
        'eq_p2': """
              dw_laplace.i.Omega2(ac.c2, q2, p2)
            - dw_dot.i.Omega2(ac.w2, q2, p2)
       + %s * dw_dot.i.GammaIn(ac.wc, q2, p2)
       + %s * dw_dot.i.Gamma0_2(ac.wc2, q2, tr(g02))
       = %s * dw_integrate.i.GammaIn(ac.wc, q2)""" % (1j, 1j, 2j * p_inc),
        'eq_dp': """
              dw_dot.i.Gamma0_1(dq0, p1)
            - dw_dot.i.Gamma0_1(dq0, tr(p2))
            - dw_dot.i.Gamma0_1(dq0, dp0)
            = 0""",
        'eq_sp': """
              dw_dot.i.Gamma0_1(sq0, p1)
            + dw_dot.i.Gamma0_1(sq0, tr(p2))
            - dw_dot.i.Gamma0_1(sq0, sp0)
            = 0""",
    }

    solvers = {
        'nls': ('nls.newton', {'i_max': 1,
                               'eps_a': 1e-6,
                               'eps_r': 1e-6,
                               'problem': 'nonlinear', })
    }

    mid_file = op.join(wdir, 'acoustics_macro_plate.py')

    solvers.update({
        'ls': ('ls.cm_pb',
               {'others': [mid_file],
                'coupling_variables': ['g01', 'g02', 'dp0', 'sp0'],
                'needs_problem_instance': True,
                })
    })

    return locals()
