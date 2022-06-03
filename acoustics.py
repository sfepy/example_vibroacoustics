# Vibroacoustics
#
# E.Rohan, V.Lukeš
# Homogenization of the vibro–acoustic transmission on periodically
# perforated elastic plates with arrays of resonators.
# https://arxiv.org/abs/2104.01367 (arXiv:2104.01367v1)
# https://doi.org/10.1016/j.apm.2022.05.040 (Applied Mathematical Modelling, 2022)
#
# compatible with SfePy 2022.1

import numpy as nm
import os
import sys
sys.path.append('.')
from sfepy.homogenization.micmac import get_homog_coefs_linear
from sfepy.base.conf import ProblemConf, get_standard_keywords
from sfepy.applications import PDESolverApp
from sfepy.base.base import Struct

wdir = os.path.dirname(__file__)

eps0 = 0.01  # size of RVE
freqs_ = [33000, 35900, 36150, 36200, 36250]
# freqs_ = (33000, 37000, 100)

p_inc = 30  # amplitude of incident wave
kr = 5  # resontator: density and elasticity multiplicator 

sound_speed = 343.0

mat_prop = {  # region: (cell_group, E, nu, rho)
    'a': (1, 1.55),  # rho0, Air
    'm': (2, 70e9, 0.34, 2700),  # Aluminium
    'c': (3, 0.1e9 / eps0**2, 0.48, 1200),  # Rubber
    'r': (4, 0.1e9 / eps0**2 * kr, 0.48, 1200 * kr),  # Rubber (rescaled)
}  # engineeringtoolbox.com

freqs = nm.linspace(*freqs_) if isinstance(freqs_, tuple) else freqs_
print('>>> frequencies: %f - %f (%d)' % (freqs[0], freqs[-1], len(freqs)))

define_args = {
    'filename_mesh': 'cell_mesh.vtk',
    'coefs_filename': 'coefs_acoustic',
    'mat_prop': mat_prop,
    'h': 0.12,
    'eps0': eps0,
    'freqs_': freqs_,
    'cell_size': 1,
}

coefs = get_homog_coefs_linear(None, None, None, regenerate=True,
                               micro_filename=os.path.join(wdir, 'acoustics_micro.py'),
                               define_args=define_args)

define_args = {
    'filename_mesh': 'cell_mesh_plate.vtk',
    'coefs_filename': 'coefs_acoustic_plate',
    'mat_prop': mat_prop,
}

coefs_plate = get_homog_coefs_linear(None, None, None, regenerate=True,
                                     micro_filename=os.path.join(wdir, 'acoustics_micro_plate.py'),
                                     define_args=define_args)


define_args = {
    'filename_mesh': 'waveguide_mesh.vtk',
    'sound_speed': sound_speed,
    'rho0': mat_prop['a'][1],
    'freqs': freqs,
    'p_inc': p_inc,
    'eps0': eps0,
    'coefs_filename': 'coefs_acoustic',
    'coefs_filename_plate': 'coefs_acoustic_plate',
}

options = Struct(conf=None, app_options=None,
                 output_filename_trunk=None, save_ebc=None,
                 save_ebc_nodes=None, save_regions=False,
                 save_regions_as_groups=False,
                 save_field_meshes=False, solve_not=False)

required, other = get_standard_keywords()

conf = ProblemConf.from_file('acoustics_macro.py',
                             required, other, define_args=define_args)

app = PDESolverApp(conf, options, 'ac_macro: ')
opts = conf.options
parametric_hook = conf.get_function(opts.parametric_hook)
app.parametrize(parametric_hook)
app()

# ./resview.py input/acoustics/paper_code/results/waveguide_mesh_w33000_p.vtk -v "270,90"
# ./resview.py input/acoustics/paper_code/results/waveguide_mesh_w33000_dp0.vtk -v "0,0" --position-vector "0,2,0" -f real.w:p0 imag.w:p1
