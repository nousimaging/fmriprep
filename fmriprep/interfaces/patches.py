# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Temporary patches
-----------------

"""

from random import randint
from time import sleep

from nipype.algorithms import confounds as nac
from numpy.linalg.linalg import LinAlgError
from nipype.interfaces.afni.base import AFNICommand, AFNICommandInputSpec
from nipype.base import (
    TraitedSpec,
    traits,
    File,
)



class RobustACompCor(nac.ACompCor):
    """
    Runs aCompCor several times if it suddenly fails with
    https://github.com/nipreps/fmriprep/issues/776

    """

    def _run_interface(self, runtime):
        failures = 0
        while True:
            try:
                runtime = super()._run_interface(runtime)
                break
            except LinAlgError:
                failures += 1
                if failures > 10:
                    raise
                start = (failures - 1) * 10
                sleep(randint(start + 4, start + 10))

        return runtime


class RobustTCompCor(nac.TCompCor):
    """
    Runs tCompCor several times if it suddenly fails with
    https://github.com/nipreps/fmriprep/issues/940

    """

    def _run_interface(self, runtime):
        failures = 0
        while True:
            try:
                runtime = super()._run_interface(runtime)
                break
            except LinAlgError:
                failures += 1
                if failures > 10:
                    raise
                start = (failures - 1) * 10
                sleep(randint(start + 4, start + 10))

        return runtime

class VolregInputSpec(AFNICommandInputSpec):
    in_file = File(
        desc="input file to 3dvolreg",
        argstr="%s",
        position=-1,
        mandatory=True,
        exists=True,
        copyfile=False,
    )
    in_weight_volume = traits.Either(
        traits.Tuple(File(exists=True), traits.Int),
        File(exists=True),
        desc="weights for each voxel specified by a file with an "
        "optional volume number (defaults to 0)",
        argstr="-weight '%s[%d]'",
    )
    out_file = File(
        name_template="%s_volreg",
        desc="output image file name",
        argstr="-prefix %s",
        name_source="in_file",
    )
    basefile = File(
        desc="base file for registration", argstr="-base %s", position=-6, exists=True
    )
    zpad = traits.Int(
        desc="Zeropad around the edges by 'n' voxels during rotations",
        argstr="-zpad %d",
        position=-5,
    )
    md1d_file = File(
        name_template="%s_md.nii.gz",
        desc="max displacement output file",
        argstr="-maxdisp1D %s",
        name_source="in_file",
        keep_extension=True,
        position=-4,
    )
    oned_file = File(
        name_template="%s.nii.gz",
        desc="1D movement parameters output file",
        argstr="-1Dfile %s",
        name_source="in_file",
        keep_extension=True,
    )
    verbose = traits.Bool(
        desc="more detailed description of the process", argstr="-verbose"
    )
    timeshift = traits.Bool(
        desc="time shift to mean slice time offset", argstr="-tshift 0"
    )
    copyorigin = traits.Bool(
        desc="copy base file origin coords to output", argstr="-twodup"
    )
    oned_matrix_save = File(
        name_template="%s.aff12.nii.gz",
        desc="Save the matrix transformation",
        argstr="-1Dmatrix_save %s",
        keep_extension=True,
        name_source="in_file",
    )
    interp = traits.Enum(
        ("Fourier", "cubic", "heptic", "quintic", "linear"),
        desc="spatial interpolation methods [default = heptic]",
        argstr="-%s",
    )


class VolregOutputSpec(TraitedSpec):
    out_file = File(desc="registered file", exists=True)
    md1d_file = File(desc="max displacement info file", exists=True)
    oned_file = File(desc="movement parameters info file", exists=True)
    oned_matrix_save = File(
        desc="matrix transformation from base to input", exists=True
    )


class Volreg(AFNICommand):
    """Register input volumes to a base volume using AFNI 3dvolreg command

    For complete details, see the `3dvolreg Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dvolreg.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> volreg = afni.Volreg()
    >>> volreg.inputs.in_file = 'functional.nii'
    >>> volreg.inputs.args = '-Fourier -twopass'
    >>> volreg.inputs.zpad = 4
    >>> volreg.inputs.outputtype = 'NIFTI'
    >>> volreg.cmdline  # doctest: +ELLIPSIS
    '3dvolreg -Fourier -twopass -1Dfile functional.1D -1Dmatrix_save functional.aff12.1D -prefix \
functional_volreg.nii -zpad 4 -maxdisp1D functional_md.1D functional.nii'
    >>> res = volreg.run()  # doctest: +SKIP

    >>> from nipype.interfaces import afni
    >>> volreg = afni.Volreg()
    >>> volreg.inputs.in_file = 'functional.nii'
    >>> volreg.inputs.interp = 'cubic'
    >>> volreg.inputs.verbose = True
    >>> volreg.inputs.zpad = 1
    >>> volreg.inputs.basefile = 'functional.nii'
    >>> volreg.inputs.out_file = 'rm.epi.volreg.r1'
    >>> volreg.inputs.oned_file = 'dfile.r1.1D'
    >>> volreg.inputs.oned_matrix_save = 'mat.r1.tshift+orig.1D'
    >>> volreg.cmdline
    '3dvolreg -cubic -1Dfile dfile.r1.1D -1Dmatrix_save mat.r1.tshift+orig.1D -prefix \
rm.epi.volreg.r1 -verbose -base functional.nii -zpad 1 -maxdisp1D functional_md.1D functional.nii'
    >>> res = volreg.run()  # doctest: +SKIP

    """

    _cmd = "3dvolreg"
    input_spec = VolregInputSpec
    output_spec = VolregOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == "in_weight_volume" and not isinstance(value, tuple):
            value = (value, 0)
        return super()._format_arg(name, trait_spec, value)