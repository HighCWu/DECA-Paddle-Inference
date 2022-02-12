# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import paddle
import datasets

from renderer import DecaInference, tensor2image

import onnxruntime as ort


def main(args):
    savefolder = args.savefolder
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector)

    # run DECA
    deca = DecaInference()

    encoder_sess = ort.InferenceSession(os.path.join(savefolder, 'deca_encoder.onnx'), providers=['CUDAExecutionProvider'])
    decoder_sess = ort.InferenceSession(os.path.join(savefolder, 'deca_decoder.onnx'), providers=['CUDAExecutionProvider'])

    testdata = testdata[0]
    expdata = expdata[0]
    encodes = []
    uv_texture_gts = []
    for i in range(3):
        if i == 0:
            data = testdata
        elif i == 1:
            data = expdata
        elif i == 2:
            data = testdata
        name = data['imagename']
        if i == 2:
            name += '_exp_from_' + expdata['imagename']
        images = data['image'][None,...]

        if i < 2:
            outputs = [o.name for o in encoder_sess.get_outputs()]
            codedict_values = encoder_sess.run(outputs, {"images": images})
            codedict_values = { k: v for k, v in zip(outputs, codedict_values) }
            encodes.append(codedict_values)
        else:
            # transfer exp code
            codedict_values = encodes[0]
            codedict_values['codedict_pose'][:,3:] = encodes[-1]['codedict_pose'][:,3:]
            codedict_values['codedict_exp'] = encodes[-1]['codedict_exp']

        decoder_input_names = [i.name for i in decoder_sess.get_inputs()]
        decoder_inputs = {}
        decoder_inputs.update({k: v for k, v in codedict_values.items() if k in decoder_input_names})
        decoder_inputs.update({
            'original_image': data['original_image'][None, ...],
            'tform': paddle.inverse(paddle.to_tensor(data['tform'][None, ...])).transpose([0,2,1]).detach().numpy()
        })

        outputs = [o.name for o in decoder_sess.get_outputs()]
        decoded_values = decoder_sess.run(outputs, decoder_inputs)
        decoded_values = { k: v for k, v in zip(outputs, decoded_values) }
        codedict = { k.split('codedict_')[-1]: paddle.to_tensor(v) for k, v in codedict_values.items() }
        opdict = { k.split('opdict_')[-1]: paddle.to_tensor(v) for k, v in decoded_values.items() if 'opdict_' in k }
        visdict = { k.split('visdict_')[-1]: paddle.to_tensor(v) for k, v in decoded_values.items() if 'visdict_' in k }
        if i == 2:
            print(opdict.get('uv_texture_gt', 233333333333))
            opdict['uv_texture_gt'] = uv_texture_gts[0]
        with paddle.no_grad():
            visdict = {
                'images': paddle.to_tensor(data['original_image'][None, ...]),
                'rendered_images': deca.render_images(codedict, opdict, visdict)
            }

        if i < 2:
            uv_texture_gts.append(opdict['uv_texture_gt'])
        
        os.makedirs(os.path.join(savefolder, name), exist_ok=True)

        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        
        if args.saveImages:
            for vis_name in ['images', 'rendered_images']:
                if vis_name not in visdict.keys():
                    continue
                cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), tensor2image(visdict[vis_name][0]))

    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--image_path', default='TestSamples/examples/bingbingwang.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-e', '--exp_path', default='TestSamples/examples/image02673.png', type=str, 
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='sf3d', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())
