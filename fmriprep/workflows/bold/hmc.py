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
Head-Motion Estimation and Correction (HMC) of BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_hmc_wf

"""

from nipype.interfaces import utility as niu, afni
from nipype.pipeline import engine as pe
from ...interfaces.mc import Volreg2ITK

from ...config import DEFAULT_MEMORY_MIN_GB


def init_bold_hmc_wf(mem_gb: float, omp_nthreads: int, name: str = 'bold_hmc_wf'):
    """
    Build a workflow to estimate head-motion parameters.

    This workflow estimates the motion parameters to perform
    :abbr:`HMC (head motion correction)` over the input
    :abbr:`BOLD (blood-oxygen-level dependent)` image.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmriprep.workflows.bold import init_bold_hmc_wf
            wf = init_bold_hmc_wf(
                mem_gb=3,
                omp_nthreads=1)

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``bold_hmc_wf``)

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    raw_ref_image
        Reference image to which BOLD series is motion corrected

    Outputs
    -------
    xforms
        ITKTransform file aligning each volume to ``ref_image``
    movpar_file
        MCFLIRT motion parameters, normalized to SPM format (X, Y, Z, Rx, Ry, Rz)
    rms_file
        Framewise displacement as measured by ``fsl_motion_outliers`` [Jenkinson2002]_.

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.confounds import NormalizeMotionParams

    #from nipype.algorithms.confounds import FramewiseDisplacement

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Head-motion parameters with respect to the BOLD reference
(transformation matrices, and six corresponding rotation and translation
parameters) are estimated before any spatiotemporal filtering using
AFNI 3dVolReg.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['bold_file', 'raw_ref_image']), name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['xforms', 'movpar_file']), name='outputnode'
    )

    # Head motion correction (hmc)

    mc = pe.Node(
        afni.Volreg(zpad=4, outputtype="NIFTI_GZ", args="-Fourier -prefix NULL -twopass"),
        name="mc",
        mem_gb=mem_gb * 3,
    )

    mc2itk = pe.Node(Volreg2ITK(), name="mcitk", mem_gb=0.05)

    normalize_motion = pe.Node(
        NormalizeMotionParams(format='AFNI'), name="normalize_motion", mem_gb=DEFAULT_MEMORY_MIN_GB
    )

    #fd = pe.Node(FramewiseDisplacement(parameter_source="AFNI"),name="fd")


    # fmt:off
    workflow.connect([
        (inputnode, mc, [('raw_ref_image', 'basefile'),
                              ('bold_file', 'in_file')]),
        (mc, mc2itk, [('oned_matrix_save', 'in_file')]),
        (mc, normalize_motion, [('oned_file', 'in_file')]),
        #(mc, fd, [('oned_file', 'in_file')]),
        #(fd, outputnode, [('out_file', 'rmsd_file')]),
        (mc2itk, outputnode, [('out_file', 'xforms')]),
        (normalize_motion, outputnode, [('out_file', 'movpar_file')]),
    ])
    # fmt:on
    return workflow
