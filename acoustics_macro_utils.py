# Vibroacoustics
#
# E.Rohan, V.Lukeš
# Homogenization of the vibro–acoustic transmission on periodically
# perforated elastic plates with arrays of resonators.
# https://arxiv.org/abs/2104.01367 (arXiv:2104.01367v1)

import os
import numpy as nm
from sfepy.base.base import Struct
from sfepy.homogenization.coefficients import Coefficients
from sfepy.discrete.fem import Mesh, FEDomain


def coefs2qp(out, coefs, nqp):
    others = {}
    for k, v in coefs.items():
        if type(v) is nm.float64:
            v = nm.array(v)

        if type(v) is not nm.ndarray:
            others[k] = v
            continue

        if k[0] == 's':
            out[k] = nm.tile(v, (nqp, 1, 1))

        else:
            if not(k in out):
                out[k] = nm.tile(v, (nqp, 1, 1))

    out.update(others)

    return out


def get_homogmat(coors, mode, pb, coefs_filename, omega=None):

    if mode == 'qp':
        nqp = coors.shape[0]
        outdir = pb.conf.options['output_dir']
        cfname = os.path.join(outdir, coefs_filename + '.h5')

        out = {}
        print('>>> coefs from: ', cfname)
        coefs_ = Coefficients.from_file_hdf5(cfname).to_dict()
        coefs = {}

        if 'omega' in coefs_ and omega is not None:
            idx = (nm.abs(coefs_['omega'] - omega)).argmin()
            rerr = nm.abs(coefs_['omega'][idx] - omega) / omega
            if rerr > 1e-3:
                raise ValueError('omega: given=%e, found=%e'
                    % (omega, coefs_['omega'][idx]))

            print('found coeficcients for w=%e' % coefs_['omega'][idx])
            del(coefs_['omega'])
        else:
            idx = 4  # magic index?

        for k, v in coefs_.items():
            if isinstance(v, nm.ndarray) and len(v.shape) == 3:
                coefs[k] = v[idx, ...]
            else:
                coefs[k] = v

        coefs2qp(out, coefs, nqp)

        transpose = [k for k, v in out.items()
                     if type(v) == nm.ndarray and (v.shape[-1] > v.shape[-2])]
        for k in transpose:
            out[k] = out[k].transpose((0, 2, 1))

        return out


def read_dict_hdf5(filename, level=0, group=None, fd=None):
    import tables as pt
    out = {}

    if level == 0:
        # fd = pt.openFile(filename, mode='r')
        fd = pt.open_file(filename, mode='r')
        group = fd.root

    for name, gr in group._v_groups.items():
        name = name.replace('_', '', 1)
        out[name] = read_dict_hdf5(filename, level + 1, gr, fd)

    for name, data in group._v_leaves.items():
        name = name.replace('_', '', 1)
        out[name] = data.read()

    if level == 0:
        fd.close()

    return out


def eval_phi(pb, state_p1, state_p2, p_inc):
    pvars = pb.create_variables(['P1', 'P2'])

    # transmission loss function: log10(|p_in|^2/|p_out|^2)
    pvars['P2'].set_data(nm.ones_like(state_p2) * p_inc**2)
    phi_In = pb.evaluate('ev_surface_integrate.5.GammaIn(P2)',
                        P2=pvars['P2'])
    pvars['P1'].set_data(state_p1**2)
    phi_Out = pb.evaluate('ev_surface_integrate.5.GammaOut(P1)',
                        P1=pvars['P1'])

    return 10.0 * nm.log10(nm.absolute(phi_In) / nm.absolute(phi_Out))


def post_process(out, pb, state, save_var0='p0'):
    rmap = {'g01': 0, 'g02': 0, 'g0': 0, 'dp0': 0, 'sp0': 0, 'p0': 0,
            'px': 1, 'p1': 1, 'p2': 2}

    for k in out.keys():
        if 'real_' in k or 'imag_' in k:
            newk = k[:4] + '.' + k[5:]
            out[newk] = out[k]
            del(out[k])

    midfn = pb.conf.filename_mesh_plate
    fname, _ = os.path.splitext(os.path.basename(midfn))
    fname = os.path.join(pb.output_dir, fname + '.h5')

    aux = []
    for k, v in read_dict_hdf5(fname)['step0'].items():
        if ('real' in k) or ('imag' in k):
            aux.append(k)
            vn = k.strip('_').split('_')
            key = '%s.%s' % tuple(vn)
            if key not in out:
                out[key] = Struct(name=v['name'].decode('ascii'),
                                    mode=v['mode'].decode('ascii'),
                                    dofs=[j.decode('ascii') for j in v['dofs']],
                                    var_name=v['varname'].decode('ascii'),
                                    shape=v['shape'],
                                    data=v['data'],
                                    dname=v['dname'])
                if 'imag' in k:
                    rmap[vn[1]] = 0

    absvars = [ii[4:] for ii in out.keys() if ii[0:4] == 'imag']
    for ii in absvars:
        if type(out['real' + ii]) is dict:
            rpart = out.pop('real' + ii)
            rdata = rpart['data']
            ipart = out.pop('imag' + ii)
            idata = ipart['data']
            dim = rdata.shape[1]

            varname = save_var0
            if dim > 1:
                aux = nm.zeros((rdata.shape[0], 1), dtype=nm.float64)

            data = rdata if dim < 2 else nm.hstack((rdata, aux))
            out['real' + ii] = Struct(name=rpart['name'],
                                      mode=rpart['mode'],
                                      dofs=rpart['dofs'],
                                      var_name=varname,
                                      data=data.copy())
            data = idata if dim < 2 else nm.hstack((idata, aux))
            out['imag' + ii] = Struct(name=ipart['name'],
                                      mode=ipart['mode'],
                                      dofs=ipart['dofs'],
                                      var_name=varname,
                                      data=data.copy())

        else:
            rpart = out['real' + ii].__dict__
            rdata = rpart['data']
            ipart = out['imag' + ii].__dict__
            idata = ipart['data']
            varname = rpart['var_name']

        absval = nm.absolute(rdata + 1j*idata)
        if rdata.shape[1] > 1:
            aux = nm.zeros((rpart['data'].shape[0], 1), dtype=nm.float64)
            absval = nm.hstack((absval, aux))

        out[ii[1:]] = Struct(name=rpart['name'],
                             mode=rpart['mode'],
                             dofs=rpart['dofs'],
                             var_name=varname,
                             data=absval.copy())

    # all plate variables as save_var0
    for k in out.keys():
        k0 = k.replace('imag.', '').replace('real.', '')
        if rmap[k0] == 0:
            out[k].var_name = save_var0

    return out


def get_region_entities(rvar, noff=0):
    reg = rvar.field.region
    mesh = reg.domain.mesh
    rnodes = reg.entities[0]
    coors = mesh.coors
    ngrp = mesh.cmesh.vertex_groups.squeeze()

    descs = mesh.descs[0]
    rcells = reg.entities[-1]
    rconn = mesh.get_conn(descs)[rcells]
    mat_ids = mesh.cmesh.cell_groups[rcells]

    remap = -nm.ones((nm.max(rnodes) + 1,), dtype=nm.int64)
    remap[rnodes] = nm.arange(rnodes.shape[0]) + noff
    rconn = remap[rconn]

    nmap = nm.where(remap >= 0)[0]

    return coors[rnodes, :], ngrp[rnodes], rconn, mat_ids, descs, nmap


def generate_plate_mesh(fname):
    dim_tab = {'3_4': '2_3', '3_8': '2_4'}

    mesh3d = Mesh.from_file(fname)
    domain = FEDomain('domain', mesh3d)
    domain.create_region('Omega1', 'cells of group 1')
    domain.create_region('Omega2', 'cells of group 2')
    gamma0 = domain.create_region('Gamma0', 'r.Omega1 *v r.Omega2', 'facet')
    cmesh = mesh3d.cmesh
    cmesh.setup_connectivity(2, 0)
    fcnd = cmesh.get_conn(2, 0)
    fcidxs = gamma0.entities[2]
    fcconn = []
    for ii in fcidxs:
        fcconn.append(fcnd.indices[fcnd.offsets[ii]:fcnd.offsets[ii + 1]])
    fcconn = nm.array(fcconn)

    remap = nm.zeros((nm.max(fcconn) + 1,), dtype=nm.int32)
    remap[fcconn] = 1
    ndidxs = nm.where(remap > 0)[0]
    remap[ndidxs] = nm.arange(len(ndidxs))
    coors2 = domain.mesh.coors[ndidxs, :]
    conn2 = remap[fcconn]
    ngrps2 = nm.ones((coors2.shape[0],))
    mids2 = nm.ones((conn2.shape[0],))

    midfn = fname[:-4] + '_plate.vtk'
    mesh2d = Mesh.from_data('2d plate', coors2, ngrps2, [conn2], [mids2],
                            [dim_tab[mesh3d.descs[0]]])
    mesh2d.write(midfn)

    return midfn
