import sys
sys.path.append('src/')
try :
    import display
except ModuleNotFoundError:
    sys.path.append('../src/')
    import display
import data_process as dp



"""
Ways of using the diplsay functions in this repo.
"""


patient = "00000"

patient_path = "/Users/enzo/Desktop/LIFPRO/Rapport/Segmentation/_patients/BraTS2021_{}/BraTS2021_{}_".format(patient, patient)

# Brain image
display.display3DCuts(dp.load(patient_path+"t1ce.nii.gz"))

# Brain image + mask (with the segmentation)
display.display3DCuts(dp.load(patient_path+"t1ce.nii.gz"), dp.load(patient_path+"seg.nii.gz"))