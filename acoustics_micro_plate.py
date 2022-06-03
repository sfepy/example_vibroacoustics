# Vibroacoustics
#
# E.Rohan, V.Lukeš
# Homogenization of the vibro–acoustic transmission on periodically
# perforated elastic plates with arrays of resonators.
# https://arxiv.org/abs/2104.01367 (arXiv:2104.01367v1)
# https://doi.org/10.1016/j.apm.2022.05.040 (Applied Mathematical Modelling, 2022)
#
# compatible with SfePy 2022.1

import os
import glob
import numpy as nm
from sfepy.discrete.fem.periodic import match_x_plane, match_y_plane
from sfepy.discrete.fem import FEDomain
from sfepy.homogenization.utils import define_box_regions
import sfepy.homogenization.coefs_base as cb
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.discrete.fem.mesh import Mesh


def get_inside_node(mesh):
    domain = FEDomain('aux', mesh)
    domain.create_region('Z', 'all', 'vertex')
    domain.create_region('S', 'vertices of surface', 'facet')
    inside = domain.create_region('Inside', 'r.Z -v r.S', 'vertex')

    return inside.entities[0][0]


###################################################################
def define(filename_mesh=None, coefs_filename=None, mat_prop=None):
    conf_dir = os.path.dirname(__file__)
    mesh = Mesh.from_file(os.path.join(conf_dir, filename_mesh))
    dim = mesh.dim
    bbox = mesh.get_bounding_box()

    options = {
        'coefs_filename': coefs_filename,
        'coefs': 'coefs',
        'requirements': 'requirements',
        'volume': {
            'value': nm.prod(bbox[1] - bbox[0]).sum(),
        },
        'output_dir': os.path.join(os.path.dirname(__file__), 'results'),
        'return_all': True,
        'multiprocessing': False,
    }

    inside = get_inside_node(mesh)

    regions = {
        'Surface': ('vertices of surface', 'facet'),
        'Aux': ('r.Left +s r.Right +s r.Top +s r.Bottom', 'facet'),
        'Smid': ('r.Surface -s r.Aux', 'facet'),
        'fix_p': ('vertex %d' % inside, 'vertex'),
    }

    regions.update(define_box_regions(dim, bbox[0], bbox[1]))

    regs = []
    if isinstance(mat_prop, dict):
        mat_C, mat_G, mat_rho = {}, {}, {}
        for k, v in mat_prop.items():
            if k == 'a' or k == 'c' or k == 'r':
                continue
            rname = 'Y' + k
            regions[rname] = 'cells of group %d' % v[0]
            E, nu, rho = v[1:]
            mat_C[rname] = stiffness_from_youngpoisson(dim, E, nu,
                                                       plane='stress')
            mat_G[rname] = E / (2*(1 + nu)) * nm.eye(dim, dtype=nm.float64)
            mat_rho[rname] = rho
            regs.append(rname)

        if len(regs) >= 2:
            regions.update({'Ymid': ' +v '.join(['r.' + ii for ii in regs])})
        elif len(regs) == 1:
            regions.update({'Ymid': 'copy r.%s' % regs[0]})
        else:
            raise NotImplementedError

    else:
        regions.update({'Ymid': 'all'})

        E, nu, rho = mat_prop
        mat_C = stiffness_from_youngpoisson(dim, E, nu, plane='stress')
        mat_G = E / (2*(1 + nu)) * nm.eye(dim, dtype=nm.float64)
        mat_rho = rho

    materials = {
        'mat_CG': ({'C': mat_C, 'G': mat_G, 'rho': mat_rho},),
        'load': ({'one': 1.0},),
    }

    fields = {
        'corrector_u': ('real', dim, 'Ymid', 1),
        'corrector_p': ('real', 1, 'Ymid', 1),
    }

    integrals = {
        'i': 2,
    }

    variables = {
        'u': ('unknown field', 'corrector_u'),
        'v': ('test field', 'corrector_u', 'u'),
        'p': ('unknown field', 'corrector_p'),
        'q': ('test field', 'corrector_p', 'p'),
        'Pi': ('parameter field', 'corrector_u', 'u'),
        'Y': ('parameter field', 'corrector_p', 'p'),
        'Pi1u': ('parameter field', 'corrector_u', '(set-to-None)'),
        'Pi2u': ('parameter field', 'corrector_u', '(set-to-None)'),
        'Pi1p': ('parameter field', 'corrector_p', '(set-to-None)'),
        'Pi2p': ('parameter field', 'corrector_p', '(set-to-None)'),
    }

    functions = {
        'match_x_plane': (match_x_plane,),
        'match_y_plane': (match_y_plane,),
    }

    ebcs = {
        'fixed_u': ('Corners', {'u.all': 0.0}),
        'fixed_p': ('fix_p', {'p.all': 0.0}),
    }

    epbcs = {
        'periodic_u_x': (['Left', 'Right'], {'u.all': 'u.all'},
                         'match_x_plane'),
        'periodic_u_y': (['Bottom', 'Top'], {'u.all': 'u.all'},
                         'match_y_plane'),
        'periodic_p_x': (['Left', 'Right'], {'p.all': 'p.all'},
                         'match_x_plane'),
        'periodic_p_y': (['Bottom', 'Top'], {'p.all': 'p.all'},
                         'match_y_plane'),
    }

    all_periodic_u = ['periodic_u_%s' % ii for ii in ['x', 'y', 'z'][:dim]]
    all_periodic_p = ['periodic_p_%s' % ii for ii in ['x', 'y', 'z'][:dim]]

    coefs = {
        'Vol_Y': {
            'regions': ['Ymid'] + regs,
            'expression': 'ev_volume.i.%s(p)',
            'class': cb.VolumeFractions,
        },
        'Cm': {
            'requires': ['pis_u', 'vchi_rs'],
            'expression': 'dw_lin_elastic.i.Ymid(mat_CG.C, Pi1u, Pi2u)',
            'set_variables': [('Pi1u', ('pis_u', 'vchi_rs'), 'u'),
                              ('Pi2u', ('pis_u', 'vchi_rs'), 'u')],
            'class': cb.CoefSymSym,
        },
        'Gm': {
            'requires': ['pis_p', 'chi_r'],
            'expression': 'dw_diffusion.i.Ymid(mat_CG.G, Pi1p, Pi2p)',
            'set_variables': [('Pi1p', ('pis_p', 'chi_r'), 'p'),
                              ('Pi2p', ('pis_p', 'chi_r'), 'p')],
            'class': cb.CoefDimDim,
        },
        'Hm': {
            'requires': ['vchi_rs', 'bvchi'],
            'expression': 'dw_lin_elastic.i.Ymid(mat_CG.C, Pi1u, Pi2u)',
            'set_variables': [('Pi1u', 'vchi_rs', 'u'),
                              ('Pi2u', 'bvchi', 'u')],
            'class': cb.CoefSym,
        },
        'Km': {
            'requires': ['bvchi'],
            'expression': 'dw_lin_elastic.i.Ymid(mat_CG.C, Pi1u, Pi2u)',
            'set_variables': [('Pi1u', 'bvchi', 'u'),
                              ('Pi2u', 'bvchi', 'u')],
            'class': cb.CoefOne,
        },
        'rhos': {
            'requires': [],
            'expression': 'ev_integrate_mat.i.Ymid(mat_CG.rho, Pi1p)',
            'set_variables': [],
            'class': cb.CoefOne,
        },
        'filenames': {},
    }

    requirements = {
        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
        },
        'pis_p': {
            'variables': ['p'],
            'class': cb.ShapeDim,
        },
        'vchi_rs': {
            'requires': ['pis_u'],
            'ebcs': ['fixed_u'],
            'epbcs': all_periodic_u,
            'equations': {
                'vchi_rs': """dw_lin_elastic.i.Ymid(mat_CG.C, v, u) =
                            - dw_lin_elastic.i.Ymid(mat_CG.C, v, Pi)""",
            },
            'set_variables': [('Pi', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_ac_mid_vchi_rs',
            'dump_variables': ['u'],
        },
        'bvchi': {
            'requires': [],
            'ebcs': ['fixed_u'],
            'epbcs': all_periodic_u,
            'equations': {
                'vchi': """dw_lin_elastic.i.Ymid(mat_CG.C, v, u) =
                           dw_surface_ltr.i.Smid(load.one, v)""",
            },
            'set_variables': [],
            'class': cb.CorrOne,
            'save_name': 'corrs_ac_mid_bvchi',
            'dump_variables': ['u'],
        },
        'chi_r': {
            'requires': ['pis_p'],
            'ebcs': ['fixed_p'],
            'epbcs': all_periodic_p,
            'equations': {
                'chi_r': """dw_diffusion.i.Ymid(mat_CG.G, q, p) =
                          - dw_diffusion.i.Ymid(mat_CG.G, q, Y)""",
            },
            'set_variables': [('Y', 'pis_p', 'p')],
            'class': cb.CorrDim,
            'save_name': 'corrs_ac_mid_chi_r',
            'dump_variables': ['p'],
        },
    }

    solvers = {
        'ls': ('ls.scipy_direct', {}),
        'newton': ('nls.newton', {'i_max': 1,
                                  'eps_a': 1e-3,
                                  'problem': 'nonlinear', }),
    }

    return locals()
