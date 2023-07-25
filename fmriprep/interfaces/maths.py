import os

import numpy as np
import nibabel as nb
from nipype.interfaces.base import File, SimpleInterface, TraitedSpec, traits, CommandLineInputSpec, InputMultiPath
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.workbench.base import WBCommand


class ClipInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Input imaging file")
    out_file = File(desc="Output file name")
    minimum = traits.Float(
        -np.inf, usedefault=True, desc="Values under minimum are set to minimum"
    )
    maximum = traits.Float(np.inf, usedefault=True, desc="Values over maximum are set to maximum")


class ClipOutputSpec(TraitedSpec):
    out_file = File(desc="Output file name")


class Clip(SimpleInterface):
    """Simple clipping interface that clips values to specified minimum/maximum

    If no values are outside the bounds, nothing is done and the in_file is passed
    as the out_file without copying.
    """

    input_spec = ClipInputSpec
    output_spec = ClipOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb

        img = nb.load(self.inputs.in_file)
        data = img.get_fdata()

        out_file = self.inputs.out_file
        if out_file:
            out_file = os.path.join(runtime.cwd, out_file)

        if np.any((data < self.inputs.minimum) | (data > self.inputs.maximum)):
            if not out_file:
                out_file = fname_presuffix(
                    self.inputs.in_file, suffix="_clipped", newpath=runtime.cwd
                )
            np.clip(data, self.inputs.minimum, self.inputs.maximum, out=data)
            img.__class__(data, img.affine, img.header).to_filename(out_file)
        elif not out_file:
            out_file = self.inputs.in_file

        self._results["out_file"] = out_file
        return runtime


class Label2MaskInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Input label file")
    label_val = traits.Int(mandatory=True, dec="Label value to create mask from")


class Label2MaskOutputSpec(TraitedSpec):
    out_file = File(desc="Output file name")


class Label2Mask(SimpleInterface):
    """Create mask file for a label from a multi-label segmentation"""

    input_spec = Label2MaskInputSpec
    output_spec = Label2MaskOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)

        mask = np.uint16(img.dataobj) == self.inputs.label_val
        out_img = img.__class__(mask, img.affine, img.header)
        out_img.set_data_dtype(np.uint8)

        out_file = fname_presuffix(self.inputs.in_file, suffix="_mask", newpath=runtime.cwd)

        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime

class CustomApplyMaskInputSpec(TraitedSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="Image to be masked")
    mask_file = File(
        exists=True,
        mandatory=True,
        desc='Mask to be applied')

class CustomApplyMaskOutputSpec(TraitedSpec):
    out_file = File(exist=True, desc="Image with mask applied")

class CustomApplyMask(SimpleInterface):
    input_spec = CustomApplyMaskInputSpec
    output_spec = CustomApplyMaskOutputSpec

    def _run_interface(self, runtime):
        #define masked output name
        out_file = fname_presuffix(
            self.inputs.in_file,
            newpath=runtime.cwd,
            suffix='_masked.nii.gz',
            use_ext=False)

        #load in input and mask
        input_img = nb.load(self.inputs.in_file)
        input_data = input_img.get_fdata()
        mask_data = nb.load(self.inputs.mask_file).get_fdata()
        #elementwise multiplication to apply mask
        out_data = input_data * mask_data
        #save out masked image and pass on file name
        nb.Nifti1Image(out_data, input_img.affine, header=input_img.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime

class SimpleMathInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Input imaging file")

class SimpleMathOutputSpec(TraitedSpec):
    out_file = File(desc="Output file name")

class BinaryMathInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Image to operate on")
    operand_file = File(exists=True, mandatory=True, desc="Image to perform operation with")
    operand_value = traits.Float(mandatory=False, desc="Value to perform operation with")

class SimpleStatsOutputSpec(TraitedSpec):
    out_stat = traits.Any(desc='stats output')

class StdDevVol(SimpleInterface):
    "Create new volume for standard deviation across time (per voxel)"

    input_spec = SimpleMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        img_data = img.get_fdata()

        std_img_data = np.std(img_data,axis=3)
        out_img = nb.Nifti1Image(std_img_data, img.affine, header=img.header)

        out_file = fname_presuffix(self.inputs.in_file, suffix="_std", newpath=runtime.cwd)

        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime
    
class MeanVol(SimpleInterface):
    "Create new volume for mean across time"

    input_spec = SimpleMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self, runtime):
        img = nb.load(self.inputs.in_file)
        img_data = img.get_fdata()

        mean_img_data = np.mean(img_data,axis=3)
        out_img = nb.Nifti1Image(mean_img_data, img.affine, header=img.header)

        out_file = fname_presuffix(self.inputs.in_file, suffix="_mean", newpath=runtime.cwd)

        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime
    
class BinaryDiv(SimpleInterface):
    "Create covariance volume"

    input_spec = BinaryMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self, runtime):
        in_img = nb.load(self.inputs.in_file)
        in_img_data = in_img.get_fdata()

        op_img = nb.load(self.inputs.operand_file)
        op_img_data = op_img.get_fdata()

        out_img_data = np.divide(in_img_data,op_img_data)

        out_img = nb.Nifti1Image(out_img_data, in_img.affine, header=in_img.header)
        out_file = fname_presuffix(self.inputs.in_file, suffix="_div", newpath=runtime.cwd)
        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime
    
class NonZeroMean(SimpleInterface):
    "Calculate mean of nonzero voxels"

    input_spec = SimpleMathInputSpec
    output_spec = SimpleStatsOutputSpec

    def _run_interface(self, runtime):
        #load in img data
        in_img = nb.load(self.inputs.in_file)
        in_img_data = in_img.get_fdata()

        #find mean of nonzeros
        in_img_data = np.array(in_img_data)
        img_mean = np.mean(in_img_data[in_img_data!=0])

        self._results["out_stat"] = img_mean
        return runtime
    
class NonZeroStDev(SimpleInterface):
    "Calculate mean of nonzero voxels"

    input_spec = SimpleMathInputSpec
    output_spec = SimpleStatsOutputSpec

    def _run_interface(self, runtime):
        #load in img data
        in_img = nb.load(self.inputs.in_file)
        in_img_data = in_img.get_fdata()

        #find mean of nonzeros
        in_img_data = np.array(in_img_data)
        img_mean = np.std(in_img_data[in_img_data!=0])

        self._results["out_stat"] = img_mean
        return runtime
    
class BinarizeVol(SimpleInterface):
    "Set values less than 0 to 0, greater than 0 to 1"

    input_spec = SimpleMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self, runtime):
        #load in img data
        in_img = nb.load(self.inputs.in_file)
        in_img_data = in_img.get_fdata()

        #set as array for easy broadcasting
        in_img_data = np.array(in_img_data)

        #perform binarization
        in_img_data[in_img_data<0] = 0
        in_img_data[in_img_data>0] = 1

        out_img = nb.Nifti1Image(in_img_data, in_img.affine, header=in_img.header)
        out_file = fname_presuffix(self.inputs.in_file, suffix="_bin", newpath=runtime.cwd)
        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime
    
class ThreshInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Image to threshold")
    thresh = traits.Float(mandatory=True, desc="Cutoff value")

class Thresh(SimpleInterface):
    "Sets values below threshold to 0"

    input_spec = ThreshInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self,runtime):
        #load img data
        in_img = nb.load(self.inputs.in_file)
        in_img_data = in_img.get_fdata()

        #set as array for easy broadcasting
        in_img_data = np.array(in_img_data)

        #apply threshold
        thresh_val = self.inputs.thresh
        in_img_data[in_img_data<thresh_val] = 0

        #write out new img
        out_img = nb.Nifti1Image(in_img_data, in_img.affine, header=in_img.header)
        out_file = fname_presuffix(self.inputs.in_file, suffix="_thr", newpath=runtime.cwd)
        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime

class WBSmoothVolInputSpec(CommandLineInputSpec):

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The input NIFTI file",
    )
    sigma = traits.Float(
        mandatory=True,
        argstr="%s",
        position=1,
        desc="the sigma for the gaussian smoothing kernel, in mm",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="%s_smooth.nii.gz",
        keep_extension=True,
        argstr="%s",
        position=2,
        desc="The output NIFTI",
    )
    fwhm = traits.Bool(
        position=3,
        argstr="-fwhm",
        desc="use fwhm kernel size instead of sigma",
    )
    roi = File(
        exists=True,
        position=4,
        argstr="-roi %s",
        desc="only smooth within ROI",
    )
    fix_zeros = traits.Bool(
        position=5,
        argstr="-fix-zeros-volume",
        desc="treat values of zero in the volume as missing data",
    )
    subvol = File(
        exists=True,
        position=6,
        argstr="-roi %s",
        desc="only smooth within subvol",
    )

class WBSmoothVol(WBCommand):

    input_spec = WBSmoothVolInputSpec
    output_spec = SimpleMathOutputSpec

    _cmd = "wb_command -volume-smoothing"

class GoodVoxMask(SimpleInterface):

    input_spec = BinaryMathInputSpec
    output_spec = SimpleMathOutputSpec

    #-bin -sub %s -mul -1'
    #binarize
    #subtract operand from input
    #multiply by -1

    def _run_interface(self, runtime):
        #load in img data
        in_img = nb.load(self.inputs.in_file)
        in_img_data = in_img.get_fdata()

        #set as array for easy broadcasting
        in_img_data = np.array(in_img_data)

        #perform binarization
        in_img_data[in_img_data<0] = 0
        in_img_data[in_img_data>0] = 1

        #load in operand data
        op_data = np.array(nb.load(self.inputs.operand_file).get_fdata())

        #subtract op from input
        sub_data = in_img_data - op_data

        #multiply by -1
        out_data = np.multiply(sub_data,-1)

        out_img = nb.Nifti1Image(out_data, in_img.affine, header=in_img.header)
        out_file = fname_presuffix(self.inputs.in_file, suffix="_bin", newpath=runtime.cwd)
        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime