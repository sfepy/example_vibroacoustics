# Vibroacoustics
#
# E.Rohan, V.Lukeš
# Homogenization of the vibro–acoustic transmission on periodically
# perforated elastic plates with arrays of resonators.
# https://arxiv.org/abs/2104.01367 (arXiv:2104.01367v1)
# https://doi.org/10.1016/j.apm.2022.05.040 (Applied Mathematical Modelling, 2022)
#
# compatible with SfePy 2022.1


from sfepy.homogenization.utils import define_box_regions
import sfepy.discrete.fem.periodic as per
import sfepy.homogenization.coefs_base as cb
import sfepy.homogenization.coefs_phononic as cp
from sfepy.homogenization.coefficients import Coefficients
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.base.base import Struct, output, get_default
from sfepy.discrete.fem.mesh import set_accuracy
import numpy as nm
import os.path as op

wdir = op.dirname(__file__)
set_accuracy(1e-7)
evp_cache = {}
evp_cache['mtxs'] = {}


def _generate_mesh2d(mesh3d, h, regs, cell_size):
    from sfepy.discrete.fem import Mesh, FEDomain

    dim_tab = {'3_4': '2_3', '3_8': '2_4'}

    domain = FEDomain('domain', mesh3d)
    for k, v in regs.items():
        domain.create_region(k, v)

    domain.create_region('S0', 'vertices in (z < %e) & (z > %e)'
        % ((h / 2. + 1e-3) * cell_size, (h / 2. - 1e-3) * cell_size), 'facet')
    reg_r = ' +v r.Yr' if 'Yr' in regs else ''
    domain.create_region('Ymi', 'r.Ym +v r.Yc' + reg_r)
    S1 = domain.create_region('S1', 'r.Ymi *f r.S0', 'facet')
    cmesh = mesh3d.cmesh
    cmesh.setup_connectivity(2, 0)
    cmesh.setup_connectivity(2, 3)
    fcnd = cmesh.get_conn(2, 0)
    fcel = cmesh.get_conn(2, 3)
    fcidxs = S1.entities[2]
    fcconn = []
    fc_foe = []
    for ii in fcidxs:
        fcconn.append(fcnd.indices[fcnd.offsets[ii]:fcnd.offsets[ii + 1]])
        fc_foe.append(fcel.indices[fcel.offsets[ii]:fcel.offsets[ii + 1]])

    fcconn = nm.array(fcconn)
    fc_foe = nm.array(fc_foe)

    matid = mesh3d.cmesh.cell_groups
    aux = nm.argmax(matid[fc_foe], axis=1)
    mids2 = matid[fc_foe[nm.arange(aux.shape[0]), aux]]
    idxs = nm.where(mids2 > 1)[0]
    mids2 = mids2[idxs]
    fcconn = fcconn[idxs, :]

    remap = nm.zeros((nm.max(fcconn) + 1,), dtype=nm.int32)
    remap[fcconn] = 1
    ndidxs = nm.where(remap > 0)[0]
    remap[ndidxs] = nm.arange(len(ndidxs))
    coors2 = domain.mesh.coors[ndidxs, :]
    conn2 = remap[fcconn]
    ngrps2 = nm.ones((coors2.shape[0],))

    mesh2d = Mesh.from_data('2d plate', coors2[:,:2], ngrps2, [conn2],
                            [mids2], [dim_tab[mesh3d.descs[0]]])

    assert(nm.all(S1.entities[0] == ndidxs))

    return mesh2d


def generate_mesh2d(mesh3d, h, regs, cell_size):
    if 'mesh2d' not in evp_cache:
        mesh2d = _generate_mesh2d(mesh3d, h, regs, cell_size)
        evp_cache['mesh2d'] = mesh2d
        mesh2d.write(mesh3d.name + '_plate.vtk')

    return evp_cache['mesh2d']


def _get_problem2d(mesh3d, h, mat_prop, cell_size, mode):
    from sfepy.discrete.fem import Mesh, FEDomain, Field
    from sfepy.terms import Term
    from sfepy.discrete.conditions import Conditions, EssentialBC
    from sfepy.discrete import (FieldVariable, Material, Integral,
                                Equation, Equations, Problem)
    from sfepy.base.conf import ProblemConf

    regs = {'Y' + k: 'cells of group %d' % v[0] for k, v in mat_prop.items()}
    mesh2d = generate_mesh2d(mesh3d, h, regs, cell_size)

    regs = {'S' + k: 'cells of group %d' % v[0] for k, v in mat_prop.items()
            if not(k == 'a')}
    domain2d = FEDomain('domain', mesh2d)
    regions = {k: domain2d.create_region(k, v) for k, v in regs.items()}
    if 'r' in mat_prop:
        regions['Si'] = domain2d.create_region('Si', 'r.Sc +v r.Sr')
    else:
        regions['Si'] = domain2d.create_region('Si', 'copy r.Sc')

    Gamma = domain2d.create_region('Gamma', 'r.Sm *v r.Si', 'facet', 'Si')

    integral = Integral('i', order=2)

    rho0 = mat_prop['a'][-1]
    gamma = h / rho0

    mat_vals = {}
    for k in ['c', 'r']:
        if k not in mat_prop:
            continue

        E, nu, rho = mat_prop[k][1:]

        mat_D = stiffness_from_youngpoisson(2, E, nu)
        mat_S = E / (2 * (1 + nu)) * gamma
        mat_rho = rho

        mat_vals.update({'Shrho0' + k: mat_S, 'rho' + k: mat_rho, 'D' + k: mat_D})

    m = Material('m', **mat_vals)

    conf = ProblemConf({
        'integrals': {'i': 2},
        'materials': {'m': (mat_vals,)},
    })

    if mode == 'laplace':
        fname, ftype, term, ename =\
            'deflection', 'scalar', 'dw_laplace(m.Shrho0%s, v, u)', 'Laplace'
    elif mode =='elasticity':
        fname, ftype, term, ename =\
            'displacement', 'vector', 'dw_lin_elastic(m.D%s, v, u)', 'Elasticity'
    else:
        raise(ValueError)

    field = Field.from_args(fname, nm.float64, ftype, regions['Si'], approx_order=1)
    u = FieldVariable('u', 'unknown', field)
    v = FieldVariable('v', 'test', field, primary_var_name='u')
    terms = [Term.new(term % k, integral, regions['S' + k], m=m, v=v, u=u)
             for k in ['c', 'r'] if k in mat_prop]

    ts = terms[0] + terms[1] if len(terms) == 2 else terms[0]
    eq1 = Equation(ename, ts)

    terms = [Term.new('dw_volume_dot(m.rho%s, v, u)' % k, integral,
             regions['S' + k], m=m, v=v, u=u)
             for k in ['c', 'r'] if k in mat_prop]

    ts = terms[0] + terms[1] if len(terms) == 2 else terms[0]
    eq2 = Equation('Volume_dot', ts)

    pb = Problem('aux_2D_%s' % mode, equations=Equations([eq1, eq2]))

    fixed = EssentialBC('Fixed', Gamma, {'u.all': 0.0})
    pb.time_update(ebcs=Conditions([fixed]))
    pb.update_materials()

    pb.conf = conf

    return pb


def get_problem2d(mesh3d, h, mat_prop, cell_size, mode=None):
    key = 'problem2d_%s' % mode
    if key not in evp_cache:
        evp_cache[key] = _get_problem2d(mesh3d, h, mat_prop, cell_size, mode)

    return evp_cache[key]


class MyEigenmomenta(cp.Eigenmomenta):
    def __init__(self, name, problem, kwargs):
        cp.Eigenmomenta.__init__(self, name, problem, kwargs)
        conf = problem.conf
        self.problem = get_problem2d(problem.domain.mesh, conf.h,
                                     conf.mat_prop, conf.cell_size,
                                     mode='elasticity')


class MySimpleEVP(cp.SimpleEVP):
    flag = 'eq_lp8'

    def __init__(self, name, problem, kwargs):
        cp.SimpleEVP.__init__(self, name, problem, kwargs)
        self.problem_3d = problem
        conf = problem.conf
        self.problem = get_problem2d(problem.domain.mesh, conf.h,
                                     conf.mat_prop, conf.cell_size,
                                     mode='elasticity')

    def get_volume(self):
        pb = self.problem_3d
        return pb.evaluate(pb.conf.options.volume['expression'])

    def __call__(self, problem=None, data=None):
        evp = cp.SimpleEVP.__call__(self, problem=problem, data=data)

        print('critical frequencies (%s): ' % self.flag)
        cf = nm.sqrt(evp.eigs)
        print(cf)

        fname = op.join(wdir, 'results', 'critical_frequencies_%s.txt' % self.flag)
        with open(fname, 'wt') as f:
            f.write('\n'.join(['%f' % k for k in cf]))

        return evp

    def prepare_matrices(self, problem):
        eq1, eq2 = problem.equations
        volume = self.get_volume()

        mtx_A = eq1.evaluate(mode='weak', dw_mode='matrix',
                             asm_obj=problem.mtx_a) / volume
        mtx_M = mtx_A.copy()
        mtx_M.data[:] = 0.0
        mtx_M = eq2.evaluate(mode='weak', dw_mode='matrix',
                             asm_obj=mtx_M) / volume

        evp_cache['mtxs']['A_' + self.flag] = mtx_A
        evp_cache['mtxs']['M_' + self.flag] = mtx_M

        return mtx_A, mtx_M, None

    def post_process(self, eigs, mtx_s_phi, data, problem):
        idxs = nm.argsort(eigs)
        eigs[:] = eigs[idxs]
        mtx_s_phi = mtx_s_phi[:, idxs]

        wn = nm.sqrt(nm.diag(nm.dot(mtx_s_phi.T * evp_cache['mtxs']['M_'  + self.flag],
                                    mtx_s_phi)))
        w = mtx_s_phi.copy()
        for ii, v in enumerate(wn):
            w[:, ii] /= v

        if self.flag == 'eq_nm6':
            return w, w
        else:
            return cp.SimpleEVP.post_process(self, eigs, w, data, problem)


class MySimpleEVP_I(MySimpleEVP):
    flag = 'eq_nm6'

    def __init__(self, name, problem, kwargs):
        cp.SimpleEVP.__init__(self, name, problem, kwargs)
        self.problem_3d = problem
        conf = problem.conf
        self.problem = get_problem2d(problem.domain.mesh, conf.h,
                                     conf.mat_prop, conf.cell_size,
                                     mode='laplace')


class MySimpleEVP_II(cp.SimpleEVP):
    flag = 'eq_nm9'
    def __call__(self, problem=None, data=None):
        self.data = data
        self.cache = {}

        evp = cp.SimpleEVP.__call__(self, problem=None, data=None)

        idxs = nm.argsort(evp.eigs)
        evp.eigs = evp.eigs[idxs]
        evp.eigs_rescaled = evp.eigs_rescaled[idxs]
        evp.eig_vectors = evp.eig_vectors[:, idxs]

        evp.solve_C = self.cache['solve_C']
        evp.mtx_R = self.cache['mtx_R']

        cf = nm.sqrt(1./evp.eigs)[::-1]
        print('critical frequencies II.: ')
        print(cf)

        fname = op.join(wdir, 'results', 'critical_frequencies_%s.txt' % self.flag)
        with open(fname, 'wt') as f:
            f.write('\n'.join(['%f' % k for k in cf]))

        return evp

    def get_volume(self):
        pb = self.problem
        return pb.evaluate(pb.conf.options.volume['expression'])

    def prepare_matrices(self, problem):
        import scipy.sparse.linalg as scsla
        equations = problem.equations
        state = problem.create_state()
        mtx = equations.eval_tangent_matrices(state(), problem.mtx_a,
                                              by_blocks=True)

        volume = self.get_volume()
        cell_size = problem.conf.cell_size
        evp1 = self.data[self.requires[0]]
        n_eigs = evp1.eigs.shape[0]
        R = mtx['B'] / volume * evp1.eig_vectors
        solve = scsla.splu(mtx['C'] / volume * cell_size).solve
        H = nm.eye(n_eigs) + nm.dot(R.T, solve(R)) / self.gamma
        E = nm.diag(evp1.eigs)

        self.cache.update({'solve_C': solve, 'mtx_R': R})

        evp_cache['mtxs']['B'] = mtx['B'] / volume
        evp_cache['mtxs']['C'] = mtx['C'] / volume
        evp_cache['mtxs']['gamma'] = self.gamma

        return H, E, None

    def post_process(self, eigs, mtx_s_phi, data, problem):
        idxs = nm.argsort(eigs)
        eigs[:] = eigs[idxs]
        mtx_s_phi = mtx_s_phi[:, idxs]

        E = nm.diag(self.data[self.requires[0]].eigs)
        wn = nm.sqrt(nm.diag(nm.dot(nm.dot(mtx_s_phi.T, E), mtx_s_phi)))
        w = mtx_s_phi.copy()
        for ii, v in enumerate(wn):
            w[:, ii] /= v
        return w, w

    def save(self, eigs, mtx_phi, problem):
        pass


class MySimpleEVP_II_rhs_3(cb.CorrMiniApp):
    flag = '_3'

    def get_volume(self):
        pb = self.problem
        return pb.evaluate(pb.conf.options.volume['expression'])

    def __call__(self, volume=None, problem=None, data=None):
        problem = get_default(problem, self.problem)

        evp1 = data[self.requires[0]]
        evp2 = data[self.requires[1]]

        problem.set_equations(self.equations)
        problem.select_bcs(ebc_names=self.ebcs, epbc_names=self.epbcs,
                           lcbc_names=self.get('lcbcs', []))
        problem.update_materials(problem.ts)

        evp_cache['ebcs' + self.flag] = problem.ebcs
        evp_cache['epbcs' + self.flag] = problem.epbcs
        state = problem.create_state()
        out = problem.equations.eval_residuals(state(), by_blocks=True)
        volume = self.get_volume()

        vec_r = out['vec_r'] / volume if 'vec_r' in out else None
        vec_b = out['vec_b'] / volume if 'vec_b' in out else None
        Cib = evp2.solve_C(vec_b)
        print('MySimpleEVP_II_rhs%s:' % self.flag)
        if vec_r is not None:
            evp_cache['mtxs']['vec_r' + self.flag] = nm.atleast_2d(vec_r).T
        if vec_b is not None:
            evp_cache['mtxs']['vec_b' + self.flag] = nm.atleast_2d(vec_b).T
            evp_cache['mtxs']['Cib' + self.flag] = Cib

        return Struct(name='rhs' + self.flag, r=vec_r, b=vec_b, Cib=Cib)


class MySimpleEVP_II_rhs_xi(MySimpleEVP_II_rhs_3):
    flag = '_xi'


class MySimpleEVP_II_rhs_a(MySimpleEVP_II_rhs_3):
    flag = '_a'

    def __call__(self, volume=None, problem=None, data=None):
        problem = get_default(problem, self.problem)

        evp1 = data[self.requires[0]]
        evp2 = data[self.requires[1]]

        vec_r, vec_b = [], []
        volume = self.get_volume()

        for ir in range(2):
            if hasattr(self, 'eq_pars'):
                eqs = {k: v % self.eq_pars[ir] for k, v in self.equations.items()}
            else:
                eqs = self.equations

            problem.set_equations(eqs)
            problem.select_bcs(ebc_names=self.ebcs, epbc_names=self.epbcs,
                               lcbc_names=self.get('lcbcs', []))
            problem.update_materials(problem.ts)

            evp_cache['ebcs' + self.flag] = problem.ebcs
            evp_cache['epbcs' + self.flag] = problem.epbcs

            if hasattr(self, 'set_variables'):
                variables = problem.get_variables()
                for (var, req, comp) in self.set_variables:
                    variables[var].set_data(data[req].states[ir][comp])

            state = problem.create_state()
            out = problem.equations.eval_residuals(state(), by_blocks=True)

            vec_ri, vec_bi = None, None
            if 'vec_r' in out:
                vec_ri = out['vec_r'] / volume
                vec_r.append(vec_ri)
            if 'vec_b' in out:
                vec_bi = out['vec_b'] / volume
                vec_b.append(vec_bi)

            Cib = evp2.solve_C(vec_b[-1])

        vec_r = nm.array(vec_r).T if len(vec_r) > 0 else None
        vec_b = nm.array(vec_b).T if len(vec_b) > 0 else None
        Cib = evp2.solve_C(vec_b)
        print('MySimpleEVP_II_rhs%s' % self.flag)
        evp_cache['mtxs']['vec_r' + self.flag] = vec_r
        evp_cache['mtxs']['vec_b' + self.flag] = vec_b
        evp_cache['mtxs']['Cib' + self.flag] = Cib

        return Struct(name='rhs' + self.flag, r=vec_r, b=vec_b, Cib=Cib)


class MySimpleEVP_II_rhs_bar(MySimpleEVP_II_rhs_a):
    flag = '_bar'


class MassTensorEval(cb.MiniAppBase):
    def __call__(self, volume=None, problem=None, data=None):
        return nm.array([*map([*data.values()][0].evaluate,
                              self.freqs)]) * self.gamma


class PhonoTensorM33(cb.MiniAppBase):
    mul_total = 1

    def __call__(self, volume=None, problem=None, data=None):
        print(self.__repr__())
        problem = get_default(problem, self.problem)

        evp1 = data[self.requires[0]]
        evp2 = data[self.requires[1]]
        rhs_1 = data[self.requires[2]]
        rhs_2 = data[self.requires[3]]

        freqs = self.freqs
        gamma = self.gamma    

        h1 = nm.dot(evp2.mtx_R.T, rhs_1.Cib) / gamma
        if rhs_1.r is not None:
            h1 += nm.dot(evp1.eig_vectors.real.T, rhs_1.r)

        h2 = nm.dot(evp2.mtx_R.T, rhs_2.Cib) / gamma
        if rhs_2.r is not None:
            h2 += nm.dot(evp1.eig_vectors.real.T, rhs_2.r)

        ZTh1 = nm.dot(evp2.eig_vectors.real.T, h1)
        ZTh2 = nm.dot(evp2.eig_vectors.real.T, h2)

        d1 = 1 if len(h1.shape) == 1 else h1.shape[1]
        d2 = 1 if len(h2.shape) == 1 else h2.shape[1]

        bTCib = nm.dot(rhs_1.b.T, rhs_2.Cib).reshape(d1, d2)

        out = []
        for w in freqs:
            E = nm.diag(w**2 / (1 - evp2.eigs * w**2))
            if not nm.isfinite(E).all():
                raise ValueError('frequency %e too close to resonance!' % w)

            out.append(bTCib
                + gamma * nm.dot(ZTh1.T, nm.dot(E, ZTh2)).reshape(d1, d2))

        return nm.array(out) * self.mul_total


class PhonoTensorMa(PhonoTensorM33):
    mul_total = 1


class PhonoTensorCD(PhonoTensorM33):
    mul_total = -1


class PhonoTensorAB(PhonoTensorM33):
    mul_total = -1


class PhonoTensorF(PhonoTensorAB):
    mul_total = 1


class PhonoTensor(cb.MiniAppBase):
    def __call__(self, volume=None, problem=None, data=None):
        C1 = data[self.requires[0]]
        C2 = data[self.requires[1]]

        n_freq, _, nc = C1.shape
        if len(self.requires) >= 3:
            C3 = data[self.requires[2]]
            nce = 1
        else:
            nce = 0

        out = nm.zeros((n_freq, 3, nc + nce), dtype=nm.float64)
        out[:, :2, :nc] = C1
        out[:, 2:3, :nc] = C2
        if nce:
            out[:, :nc, 2:3] = C2.transpose([0, 2, 1])
            out[:, 2:3, 2:3] = C3

        return out


##################################################################
def define(filename_mesh=None, coefs_filename=None, mat_prop=None,
           h=None, eps0=None, freqs_=None, cell_size=1):

    freqs = nm.linspace(*freqs_) if isinstance(freqs_, tuple)\
        else nm.array(freqs_)

    eps = 1e-3
    fcp = h / 2. * (1 + eps) * cell_size
    fcm = h / 2. * (1 - eps) * cell_size
    regions = {
        'Y': 'all',
        'Ys': ('r.Ym +v r.Yi', 'cell'),
        'S1': ('vertices in (z < %e) & (z > %e)' % (fcp, fcm), 'facet'),
        'S2': ('vertices in (z < %e) & (z > %e)' % (-fcm, -fcp), 'facet'),
        'Si1': ('r.Yi *v r.S1', 'facet', 'Ya'),
        'Si2': ('r.Yi *v r.S2', 'facet', 'Ya'),
        'Ss1': ('r.Ys *v r.S1', 'facet', 'Ya'),
        'Ss2': ('r.Ys *v r.S2', 'facet', 'Ya'),
        'Gamma_sa': ('r.Ys *f r.Ya', 'facet', 'Ya'),
        'Gamma_cm': ('r.Ym *v r.Yc', 'facet', 'Yc'),
        'Gamma_Scm': ('r.Gamma_cm *v r.Si1', 'vertex', 'Si1'),
        'Ip': ('r.Ya *v r.Top', 'facet', 'Ya'),
        'Im': ('r.Ya *v r.Bottom', 'facet', 'Ya'),
        'Ya_left': ('r.Left *v r.Ya', 'facet', 'Ya'),
        'Ya_right': (' r.Right *v r.Ya', 'facet', 'Ya'),
        'Ya_top': ('r.Top *v r.Ya', 'facet', 'Ya'),
        'Ya_bottom': ('r.Bottom *v r.Ya', 'facet', 'Ya'),
        'Ya_far': ('r.Far *v r.Ya', 'facet', 'Ya'),
        'Ya_near': ('r.Near *v r.Ya', 'facet', 'Ya'),
        'Ya_fix': ('vertices of group 1', 'vertex'),
    }

    regions.update(define_box_regions(3, nm.array((-.5, -.5, -.5)) * cell_size,
                                         nm.array((.5, .5, .5)) * cell_size))

    for k, v in mat_prop.items():
        regions['Y' + k] = 'cells of group %d' % v[0]
        regions['S%s1' % k] = ('r.Y%s *v r.S1' % k, 'facet', 'Ya')

    regions['Yi'] = 'r.Yc +v r.Yr' if 'r' in mat_prop else 'copy r.Yc'

    fields = {
        'displacement': ('real', 'vector', 'Si1', 1),
        'deflection': ('real', 'scalar', 'Si1', 1),
        'pressure': ('real', 'scalar', 'Ya', 1),
        'aux_field': ('real', 'scalar', 'Y', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'w': ('unknown field', 'deflection'),
        'z': ('test field', 'deflection', 'w'),
        'W1': ('parameter field', 'deflection', '(set-to-None)'),
        'W2': ('parameter field', 'deflection', '(set-to-None)'),
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'Pi': ('parameter field', 'pressure', '(set-to-None)'),
        'P1': ('parameter field', 'pressure', '(set-to-None)'),
        'P2': ('parameter field', 'pressure', '(set-to-None)'),
        'V': ('parameter field', 'aux_field', '(set-to-None)'),
    }

    ebcs = {
        'fixed_Ya_p': ('Ya_fix', {'p.0': 0}),
        'fixed_Gamma_Scm_w': ('Gamma_Scm', {'w.0': 0}),
        'fixed_Gamma_Scm_u': ('Gamma_Scm', {'u.all': 0}),
    }

    epbcs = {}

    per_tab = {
        3: [('x', 'left', 'right'), ('y', 'near', 'far'), ('z', 'bottom', 'top')],
    }

    for d, p1, p2 in per_tab[3][:2]:
        epbcs.update({
            'per_p_' + d: (['Ya_' + p1, 'Ya_' + p2], {'p.0': 'p.0'},
                           'match_%s_plane' % d),
        })

    all_periodic_p = ['per_p_%s' % ii for ii in ['x', 'y']]

    rho0 = mat_prop['a'][-1]  # air: cell group 1

    materials = {'reg': ({'rho': {'Y' + k: v[-1] for k, v in mat_prop.items()}},)}
    materials.update({'reg_' + k: ({'rho': v[-1]},) for k, v in mat_prop.items()})

    materials['aux'] = ({
        'n1': nm.array([[1, 0, 0]]).T,
        'n2': nm.array([[0, 1, 0]]).T,
        'n3': nm.array([[0, 0, 1]]).T,
        'k': nm.eye(3),
    },)

    functions = {
        'match_x_plane': (per.match_x_plane,),
        'match_y_plane': (per.match_y_plane,),
        'match_z_plane': (per.match_z_plane,),
    }

    gamma = h / rho0

    exp_eigenmomenta = ' + '.join(['ev_integrate.i.S%s(m.rho%s, u)' % (k, k)
                                   for k in ['c', 'r'] if k in mat_prop])

    dv_regs_to_mat = {'S%s1' % k: ('reg_%s' % k, 'rho')
                      for k in ['m', 'c', 'r'] if k in mat_prop}

    vols = ['Y', 'Ys', 'Yi'] + ['Y' + k for k in mat_prop.keys()]
    surfs = ['Si1', 'Ss1', 'Ip'] + ['S%s1' % k for k in mat_prop.keys()]

    evp_solver = {
        'eigensolver': 'eig.sgscipy',
    }

    coefs = {
        'volumes': {
            'regions': vols,
            'expression': 'ev_volume.5.%s(V)',
            'class': cb.VolumeFractions,
        },
        'surfaces': {
            'regions': surfs,
            'expression': 'ev_volume.5.%s(V)',
            'class': cb.VolumeFractions,
        },
        'Vol_Imp': {
            'regions': ['Im', 'Ip'],
            'expression': 'ev_volume.5.%s(V)',
            'class': cb.VolumeFractions,
        },
        'h': {
            'requires': [],
            'expression': '%e' % h,
            'class': cb.CoefEval,
        },
        'dv_info': {
            'status': 'auxiliary',
            'requires': ['c.surfaces'],
            'region_to_material': dv_regs_to_mat,
            'class': cp.DensityVolumeInfo,
        },
        'eigenmomenta': {
            'status': 'auxiliary',
            'requires': ['evp', 'c.dv_info'],
            'expression': exp_eigenmomenta,
            'options': {
                'var_name': 'u',
                'threshold': 1e-2,
                'threshold_is_relative': True,
            },
            'class': MyEigenmomenta,
        },
        'tM_fun': {
            'status': 'auxiliary',
            'requires': ['evp', 'c.dv_info', 'c.eigenmomenta'],
            'class': cp.AcousticMassTensor,
        },
        'tM': {
            'requires': ['c.tM_fun'],
            'freqs': freqs,
            'gamma': gamma,
            'class': MassTensorEval,
        },
        'M33_freq': {
            'status': 'auxiliary',
            'requires': ['evp1', 'evp2', 'rhs_3', 'rhs_3'],
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorM33,
        },
        'M33': {
            'status': 'auxiliary',
            'requires': ['c.dv_info', 'c.M33_freq'],
            'expression': 'c.dv_info.average_density * %e + c.M33_freq' % gamma,
            'class': cb.CoefEval,
        },
        'M3a': {
            'status': 'auxiliary',
            'requires': ['evp1', 'evp2', 'rhs_3', 'rhs_a'], # !!! (a, a)?
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorMa,
        },
        'Mab': {
            'status': 'auxiliary',
            'requires': ['evp1', 'evp2', 'rhs_a', 'rhs_a'],
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorMa,
        },
        'M': {
            'requires': ['c.Mab', 'c.M3a', 'c.M33'],
            'class': PhonoTensor,
        },
        'Dab': {
            'status': 'auxiliary',
            'requires': ['evp1', 'evp2', 'rhs_a', 'rhs_bar'],
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorCD,
        },
        'D3a': {
            'status': 'auxiliary',
            'requires': ['evp1', 'evp2', 'rhs_3', 'rhs_bar'],
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorCD,
        },
        'D': {
            'requires': ['c.Dab', 'c.D3a'],
            'class': PhonoTensor,
        },
        'C3': {
            'status': 'auxiliary',
            'requires': ['evp1', 'evp2', 'rhs_3', 'rhs_xi'],
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorCD,
        },
        'Ca': {
            'status': 'auxiliary',
            'requires': ['evp1', 'evp2', 'rhs_a', 'rhs_xi'],
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorCD,
        },
        'C': {
            'requires': ['c.Ca', 'c.C3'],
            'class': PhonoTensor,
        },
        'A_freq': {
            'status': 'auxiliary',          
            'requires': ['evp1', 'evp2', 'rhs_bar', 'rhs_bar'],
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorAB,
        },
        'A': {
            'requires': ['c.volumes', 'c.surfaces', 'c.A_freq'],
            'expression':
                "c.volumes['volume_Ya'] / c.surfaces['volume_Ip']"
                " * nm.eye(2) / %e + c.A_freq" % cell_size,
            'class': cb.CoefEval,
        },
        'B': {
            'requires': ['evp1', 'evp2', 'rhs_xi', 'rhs_bar'],
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorAB,
        },
        'F': {
            'requires': ['evp1', 'evp2', 'rhs_xi', 'rhs_xi'],
            'freqs': freqs,
            'gamma': gamma,
            'class': PhonoTensorF,
        },
        'omega': {
            'expression': 'self.problem.conf.freqs',
            'class': cb.CoefEval,
        },
    }

    eq_evp = {
        'M,v,w': 'dw_dot.i.Si1(reg.rho, v, w)',
        'C,q,p': 'dw_laplace.i.Ya(q, p)',
        'BT,v,p': """dw_dot.i.Si1(v, p) 
                   - dw_dot.i.Si1(v, tr(Si2, p))""",
        'B,q,w': """dw_dot.i.Si1(q, w)
                  - dw_dot.i.Si2(q, tr(Si1, w))""",
    }

    requirements = {
        'evp': {
            'equations': None,
            'class': MySimpleEVP,
            'options': evp_solver,
            'save_name': 'evp',
            'dump_variables': ['u'],
        },
        'evp1': {
            'equations': None,
            'class': MySimpleEVP_I,
            'options': evp_solver,
            'save_name': 'evp1',
            'dump_variables': ['u'],
        },
        'evp2': {
            'requires': ['evp1'],
            'ebcs': ['fixed_Gamma_Scm_w', 'fixed_Ya_p'],
            'epbcs': all_periodic_p,
            'equations': eq_evp,
            'gamma': gamma,
            'class': MySimpleEVP_II,
            'options': evp_solver,
            'save_name': 'evp2',
            'dump_variables': ['w', 'p'],
        },
        'pis': {
            'variables': ['p'],
            'class': cb.ShapeDim,
        },
        'rhs_3': {
            'requires': ['evp1', 'evp2'],
            'ebcs': ['fixed_Gamma_Scm_w', 'fixed_Ya_p'],
            'epbcs': all_periodic_p,
            'gamma': gamma,
            'equations': {
                'vec_r,z,': 'dw_integrate.i.Si1(reg.rho, z)',
                'vec_b,q,': """dw_integrate.i.Ss1(q)
                             - dw_integrate.i.Ss2(q)"""
            },
            'class': MySimpleEVP_II_rhs_3,
            'dump_variables': ['w', 'p'],
        },
        'rhs_a': {
            'requires': ['evp1', 'evp2', 'rhs_xi', 'rhs_3', 'rhs_bar'],
            'ebcs': ['fixed_Gamma_Scm_w', 'fixed_Ya_p'],
            'epbcs': all_periodic_p,
            'gamma': gamma,
            'equations': {
                # [n^k]_dS = - [n^k]_dY
                'vec_b,q,': '-dw_surface_ndot.i.Gamma_sa(aux.n%s, q)',
            },
            'eq_pars': ['1', '2'],
            'class': MySimpleEVP_II_rhs_a,
            'dump_variables': ['w', 'p'],
        },
        'rhs_xi': {
            'requires': ['evp1', 'evp2'],
            'ebcs': ['fixed_Gamma_Scm_w', 'fixed_Ya_p'],
            'epbcs': all_periodic_p,
            'gamma': gamma,
            'equations': {
                'vec_b,q,': """dw_integrate.i.Ip(q)
                             - dw_integrate.i.Im(q)"""
            },
            'class': MySimpleEVP_II_rhs_xi,
            'dump_variables': ['w', 'p'],
        },
        'pis': {
            'variables': ['p'],
            'class': cb.ShapeDim,
        },
        'rhs_bar': {
            'requires': ['evp1', 'evp2', 'pis'],
            'ebcs': ['fixed_Gamma_Scm_w', 'fixed_Ya_p'],
            'epbcs': all_periodic_p,
            'gamma': gamma,
            'equations': {
                'vec_b,q,': 'dw_laplace.i.Ya(q, Pi)',
            },
            'set_variables': [('Pi', 'pis', 'p')],
            'class': MySimpleEVP_II_rhs_bar,
            'dump_variables': ['w', 'p'],
        },
    }

    options = {
        'coefs': 'coefs',
        'requirements': 'requirements',
        'volume': {'expression': 'ev_volume.i.Ip(p)'},
        # 'multiprocessing': True,
        'multiprocessing': False,
        'coefs_filename': coefs_filename,
        'output_dir': op.join(wdir, 'results'),
        'file_per_var': True,
        'return_all': True,
    }

    integrals = {
        'i': 2,
    }

    solvers = {
        'ls': ('ls.mumps', {}),
        'newton': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-4,
            'eps_r': 1e-4,
            'problem': 'nonlinear',
        })
    }

    return locals()
