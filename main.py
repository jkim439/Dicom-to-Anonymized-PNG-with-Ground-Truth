__author__ = 'Junghwan Kim'
__copyright__ = 'Copyright 2016-2019 Junghwan Kim. All Rights Reserved.'
__version__ = '1.0.0'

import base64
import cv2
import logging
import numpy as np
import os
import pydicom
import scipy.misc
import shutil
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from shutil import copyfile


def toRun(path, log, pkey, fernet):

    # Check folder
    if not path[-8:].isdigit():
        return
    print '\n[LOAD]', path

    # Initialize accession number
    global accessionNumber, startNumber, result
    accessionNumber = ''

    # Append list
    list_dcm = []
    list_roi = []
    list_tif = []
    for paths, dirs, files in os.walk(path):
        for name in sorted(files):
            if name.endswith('.dcm'):
                list_dcm.append(name)
            if name.endswith('.rois_series'):
                list_roi.append(name)
            if name.endswith('.tif'):
                list_tif.append(name)

    '''
    ######################################## PART I. Anonymize DICOM ########################################
    '''
    i = 0
    list_instanceNumber = []
    for i in range(len(sorted(list_dcm))):
        path_full = path + '/' + list_dcm[i]

        # Get dicom data set
        ds = pydicom.dcmread(path_full)

        # Check SpecificCharacterSet
        if 'SpecificCharacterSet' in ds:
            if ds.SpecificCharacterSet == 'ISO IR 149':
                print '[ERROR] LookupError: unknown encoding: ISO IR 149:', path_full
                raw_input('Press Enter to skip this file... ')
                continue

        # Initiate basic information
        patientid = fernet.encrypt(str(ds.PatientID))

        # Check AcquisitionTime
        if 'AcquisitionTime' in ds:
            acquisitionTime = ds.AcquisitionTime[0:6]
        else:
            acquisitionTime = '000000'

        # Generate accession number
        if 'AcquisitionDate' in ds:
            year = ds.AcquisitionDate[0:4]
            if not accessionNumber:
                accessionNumber = '%8s%0.6s%6s' % (ds.AcquisitionDate, acquisitionTime,
                                                   str(uuid.uuid4().int >> 64)[0:6])
        elif 'StudyDate' in ds:
            year = ds.StudyDate[0:4]
            if not accessionNumber:
                accessionNumber = '%8s%0.6s%6s' % (ds.StudyDate, acquisitionTime,
                                                   str(uuid.uuid4().int >> 64)[0:6])
        else:
            print '[ERROR] Not exists ds.AcquisitionDate and ds.StudyDate:', os.path.join(paths, name)
            raw_input('Press Enter to skip this file... ')

        # Generate dicom file name
        if 'InstanceNumber' in ds:
            number = '{:03d}'.format(ds.InstanceNumber)
            name_encrypted = accessionNumber + '_' + number + '.dcm'
        else:
            print '[ERROR] Not exists ds.InstanceNumber:', os.path.join(paths, name)
            raw_input('Press Enter to skip this file... ')

        # Append instanceNumber
        list_instanceNumber.append(int(ds.InstanceNumber))

        # Calculate patient age
        try:
            if ds.PatientAge == '000Y':
                age = '{:03d}Y'.format(int(year) - int(ds.PatientBirthDate[0:4]) - 1)
                ds.PatientAge = age
        except AttributeError:
            age = '{:03d}Y'.format(int(year) - int(ds.PatientBirthDate[0:4]) - 1)
            ds.add_new(0x00101010, 'AS', age)
        except TypeError:
            print '[ERROR] Incorrect ds.PatientAge type:', os.path.join(paths, name)
            raw_input('Press Enter to skip this file... ')
        except ValueError:
            ds.PatientAge = '000Y'
            print '[WARNING] Overwrite ds.PatientAge value = 000Y:', os.path.join(paths, name)

        # Write dicom data
        def write(element, value):
            try:
                attribute = getattr(ds, element)
                if attribute:
                    setattr(ds, element, value)
            except AttributeError:
                pass
            except TypeError:
                print '[ERROR] Incorrect ' + element + ' type:', os.path.join(paths, name)
                raw_input('Press Enter to skip this file... ')
            except ValueError:
                print '[ERROR] Incorrect ' + element + ' value:', os.path.join(paths, name)
                raw_input('Press Enter to skip this file... ')

        # Overwrite element
        write('AccessionNumber', accessionNumber)
        write('PatientID', patientid)

        # Clear element
        write('ContentDate', '19010101')
        write('ContentTime', '000000')
        write('DeviceSerialNumber', '0')
        write('InstitutionName', 'institution')
        write('InstitutionalDepartmentName', 'department')
        write('Manufacturer', 'manufacturer')
        write('ManufacturerModelName', 'model')
        write('OperatorsName', 'operator')
        write('PatientBirthDate', '19010101')
        write('OtherPatientIDs', '00000000')
        write('PatientName', 'Anonymous')
        write('ReferringPhysicianName', 'physician')
        write('SeriesDate', '19010101')
        write('SeriesTime', '000000')
        write('SoftwareVersions', '1.0')
        write('StationName', 'station')
        write('StudyDate', '19010101')
        write('StudyDescription', '00000000')
        write('StudyID', '1')
        write('StudyTime', '000000')
        write('InstitutionAddress', 'address')
        write('OtherPatientNames', 'Anonymous')
        write('InstanceCreationDate', '19010101')
        write('InstanceCreationTime', '000000')
        write('PerformingPhysicianName', 'phycian')
        write('NameofPhysiciansReadingStudy', 'physician')
        write('PhysiciansofRecord', 'physician')
        write('PatientWeight', '0')
        write('PatientSize', '0')
        write('PatientAddress', 'address')
        write('AdditionalPatientHistory', '')
        write('EthnicGroup', 'ethnicity')
        write('ReviewDate', '19010101')
        write('ReviewTime', '000000')
        write('ReviewerName', 'anonymous')

        path_parent = os.path.abspath(os.path.join(path, '..'))
        path_new = path_parent + '/' + accessionNumber
        folder_name = os.path.basename(path)

        # Get startNumber
        startNumber = min(list_instanceNumber)

        # Create folder at the first time
        if not os.path.exists(path_new):
            os.makedirs(path_new)
            os.makedirs(path_new + '/dcms')
            os.makedirs(path_new + '/gtruth')
            os.makedirs(path_new + '/images')
            os.makedirs(path_new + '/labels')
            os.makedirs(path_new + '/rois')
            log.write(folder_name[:7] + '\t\t' + accessionNumber + '\t\t' + pkey + '\n')

        ds.save_as(os.path.join(path_new + '/dcms', name))
        os.rename(os.path.join(path_new + '/dcms', name), os.path.join(path_new + '/dcms', name_encrypted))
        print '[SUCCESS] File processed:', os.path.join(path_new, name_encrypted)

        '''
        ######################################## PART II. DCM to PNG ########################################
        '''
        ds_array = ds.pixel_array
        intercept = ds.RescaleIntercept
        slope = ds.RescaleSlope
        ds_array = ds_array * slope + intercept
        ds_array = GetLUTValue(ds_array, 100, 50)
        AdjImage = scipy.misc.toimage(ds_array)
        AdjImage.save(os.path.join(path_new + '/images/', name_encrypted[:-4] + '.png'))

        i += 1

    '''
    ######################################## PART III. Move files ########################################
    '''
    i = 0
    for i in range(len(sorted(list_roi))):
        path_full = path + '/' + list_roi[i]
        path_parent = os.path.abspath(os.path.join(path, '..'))
        path_new = path_parent + '/' + accessionNumber

        copyfile(path_full, path_new + '/rois/' + list_roi[i])
        i += 1

    i = 0
    for i in range(len(sorted(list_tif))):
        path_full = path + '/' + list_tif[i]
        path_parent = os.path.abspath(os.path.join(path, '..'))
        path_new = path_parent + '/' + accessionNumber

        copyfile(path_full, path_new + '/rois/' + list_tif[i])
        i += 1

    '''
    ######################################## PART IV. Ground Truth ########################################
    '''
    path_parent = os.path.abspath(os.path.join(path, '..'))
    path_new = path_parent + '/' + accessionNumber

    # Get png list
    list_png = []
    for paths, dirs, files in os.walk(path_new + '/images'):
        for name in sorted(files):
            if name.endswith('.png'):
                list_png.append(name)

    # Count png list and tif list
    if len(list_png) != len(list_tif):
        print '[ERROR] The number of tif and the number of png are different.', path_new
        exit(1)

    # Ground Truth
    for png, tif in zip(sorted(list_png), sorted(list_tif)):

        img_1 = cv2.imread(path_new + '/images/' + png)
        img_1_height, img_1_width, img_1_channels = img_1.shape

        img_2 = cv2.imread(path_new + '/rois/' + tif)
        img_2_height, img_2_width, img_2_channels = img_2.shape

        # Check image size
        if img_1_height != img_2_height or img_1_width != img_2_width or img_1_channels != img_2_channels:
            print '[ERROR] The size of PNG file and the size of TIF file are different: ' + png + ', ' + tif
            exit(1)

        img = cv2.addWeighted(img_1, 1, img_2, 1, 0)
        cv2.imwrite(path_new + '/gtruth/' + png, img)

    '''
    ######################################## PART V. Move folders ########################################
    '''

    # Get label list
    list_lbl = []
    for paths, dirs, files in os.walk(path + '/labels'):
        for name in sorted(files):
            if name.endswith('.png'):
                list_lbl.append(name)

    j = startNumber
    i = 0

    for i in range(len(sorted(list_lbl))):
        number = '{:03d}'.format(j)
        name_encrypted = accessionNumber + '_' + number + '.png'

        path_full = path + '/labels/' + list_lbl[i]
        path_parent = os.path.abspath(os.path.join(path, '..'))
        path_new = path_parent + '/' + accessionNumber

        copyfile(path_full, path_new + '/labels/' + name_encrypted)
        j += 1
        i += 1


    if os.path.exists(path + '/log'):
        shutil.copytree(path + '/log', path_new + '/log')

    result += 1

    return None


def GetLUTValue(data, window, level):
    """Apply the RGB Look-Up Table for the given data and window/level value."""

    lutvalue = np.piecewise(data,
                            [data <= (level - 0.5 - (window - 1) / 2),
                             data > (level - 0.5 + (window - 1) / 2)],
                            [0, 255, lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0)])
    # Convert the resultant array to an unsigned 8-bit array to create
    # an 8-bit grayscale LUT since the range is only from 0 to 255
    return np.array(lutvalue, dtype=np.uint8)


# Set path
path = '/home/jkim/KMS/SET2'
folder = os.listdir(path)
global result
result = 0

# LOG file
log = open(path + '/log.log', 'w')
log.write('PID\t\tAccession Number\t\tPKEY\n-------\t\t--------------------\t\t--------------------------------------------\n')

# PKEY file
kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=os.urandom(16), iterations=100000, backend=default_backend())
pkey = base64.urlsafe_b64encode(kdf.derive('aksdjlasdjlasjdlas'))
fernet = Fernet(pkey)

# Recur load folder function
i = 0
while i < len(folder):
    if not folder[i].endswith('.log'):
        toRun(path + '/' + folder[i], log, pkey, fernet)
    i += 1

# Print result
print '\n----------------------------------------------------------------------------------------------------' \
      '\nResult' \
      '\n----------------------------------------------------------------------------------------------------' \
      '\n', result, 'Folders are processed successfully.'
