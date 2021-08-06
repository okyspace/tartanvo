from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO

import argparse
import numpy as np
import cv2
from os import mkdir
from os.path import isdir
import torch
import torch.nn as nn

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--tartanair', action='store_true', default=False,
                        help='tartanair test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--results', default='results/',
                        help='results path (default: results/)')
    parser.add_argument('--scene', default='',
                        help='scene (default: )')
    parser.add_argument('--traj-num', default='',
                        help='traj num (default: )')
    parser.add_argument('--difficulty', default='',
                        help='difficulty (default: )')
    parser.add_argument('--threshold-error-avg', type=float, default=0.1,
                        help='frame error avg threshold (default: 0.1)')
    parser.add_argument('--threshold-error-trans', type=float, default=0.2,
                        help='frame error trans threshold (default: 0.2)')
    parser.add_argument('--threshold-error-rot', type=float, default=0.025,
                        help='frame error rot threshold (default: 0.025)')
    args = parser.parse_args()

    return args

def save_to_file(folder, scene, difficulty, traj_num, num_poses, ate, rpe, thres_avg, thres_trans, thres_rot, frames_error_avg, frames_exceeded_avg_thres, frames_error_trans, frames_exceeded_trans_thres, frames_error_rot, frames_exceeded_rot_thres):
    # save ate, rpe, frame error to file
    f = open(folder + '/eval_' + traj_num + '.txt', 'w')

    # save basic traj info
    f.write('Scene: ' + scene + "\n")
    f.write('Difficulty: ' + difficulty + "\n")
    f.write('Trajectory Number: ' + traj_num + "\n")
    f.write("Total num of frames: " + str(num_poses) + "\n")

    # save ATE and RPE
    f.write('ATE: ' +  str(ate) + "\n")
    f.write('RPE: ' +  str(rpe) + "\n\n")

    # save frame errors (Avg)
    f.write('====================== Frames RMSE [Avg] ======================\n')
    f.write(str(np.around(frames_error_avg, 4)) + "\n\n")     
    # save frame idxs if error > threshold
    f.write("No. of frames that exceeded threshold: " + str(frames_exceeded_avg_thres.shape[0]) + "\n")
    f.write("Index of frames with RMSE greater than threshold: " + str(thres_avg) + "\n")
    f.write(str(frames_exceeded_avg_thres) + "\n\n")

    # save frame errors (translation)
    f.write('====================== Frames RMSE [Translation] ======================\n')
    frames_error_trans = np.around(frames_error_trans, 4)
    # frames_error_trans[frames_exceeded_trans_thres] = -1
    f.write(str(frames_error_trans) + "\n\n")     

    # save frame idxs if error > threshold
    f.write("No. of frames that exceeded threshold: " + str(frames_exceeded_trans_thres.shape[0]) + "\n")
    f.write("Index of frames with RMSE greater than threshold: " + str(thres_trans) + "\n")
    f.write(str(frames_exceeded_trans_thres) + "\n\n")

    # save frame errors (rotational)
    f.write('====================== Frames RMSE [Rotational] ======================\n')
    f.write(str(np.around(frames_error_rot, 4)) + "\n\n")     
    # save frame idxs if error > threshold
    f.write("No. of frames that exceeded threshold: " + str(frames_exceeded_rot_thres.shape[0]) + "\n")
    f.write("Index of frames with RMSE greater than threshold: " + str(thres_rot) + "\n")
    f.write(str(frames_exceeded_rot_thres) + "\n\n")

    
    f.write('############################ Raw ############################')
    # save frame errors (Avg)
    f.write('====================== Frames RMSE [Avg] ======================\n')
    f.write(str(np.around(frames_error_avg, 4)) + "\n\n")     
    # save frame idxs if error > threshold
    f.write("No. of frames that exceeded threshold: " + str(frames_exceeded_avg_thres.shape[0]) + "\n")
    f.write("Index of frames with RMSE greater than threshold: " + str(thres_avg) + "\n")
    f.write(str(frames_exceeded_avg_thres) + "\n\n")

    # save frame errors (translation)
    f.write('====================== Frames RMSE [Translation] ======================\n')
    f.write(str(np.around(frames_error_trans, 4)) + "\n\n")     

    # save frame idxs if error > threshold
    f.write("No. of frames that exceeded threshold: " + str(frames_exceeded_trans_thres.shape[0]) + "\n")
    f.write("Index of frames with RMSE greater than threshold: " + str(thres_trans) + "\n")
    f.write(str(frames_exceeded_trans_thres) + "\n\n")

    # save frame errors (rotational)
    f.write('====================== Frames RMSE [Rotational] ======================\n')
    f.write(str(np.around(frames_error_rot, 4)) + "\n\n")     
    # save frame idxs if error > threshold
    f.write("No. of frames that exceeded threshold: " + str(frames_exceeded_rot_thres.shape[0]) + "\n")
    f.write("Index of frames with RMSE greater than threshold: " + str(thres_rot) + "\n")
    f.write(str(frames_exceeded_rot_thres) + "\n\n")
    f.close()


def compute_frame_error(predicted, gt):
    """
    calculate loss between GT and predicted
    """
    predicted_trans = predicted[:, :3]
    predicted_rot = predicted[:, 3:]
    gt_trans = gt.numpy()[:, :3]
    gt_rot = gt.numpy()[:, 3:]

    mse_trans = ((predicted_trans - gt_trans) ** 2).mean(axis=1)
    rmse_trans = np.sqrt(mse_trans)

    mse_rot = ((predicted_rot - gt_rot) ** 2).mean(axis=1)
    rmse_rot = np.sqrt(mse_rot)

    avg_rmse = (rmse_trans + rmse_rot) / 2.0

    return avg_rmse, rmse_trans, rmse_rot



if __name__ == '__main__':
    args = get_args()

    # get vo model
    testvo = TartanVO(args.model_name)

    # load trajectory data from a folder
    datastr = 'tartanair'
    if args.kitti:
        datastr = 'kitti'
    elif args.euroc:
        datastr = 'euroc'
    else:
        datastr = 'tartanair'
    print("datastr: {}".format(datastr))

    # get cam intrinsics
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr=='kitti':
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    testDataset = TrajFolderDataset(args.test_dir,  posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    # get model name
    model_name = args.model_name.split('.')[0]
    model_name = model_name.split('/')[-1]
    print("model name: ", model_name)

    # testname = datastr + '_' + args.model_name.split('.')[0]
    testname = datastr + '_' + model_name
    print("testname: ", testname)

    # for optical flow
    if args.save_flow:
        flowdir = args.results+testname+'_flow'
        if not isdir(flowdir):
            mkdir(flowdir)
        flowcount = 0

    #     
    motionlist = []
    frames_avg_error = []
    frames_trans_error = []
    frames_rot_error = []
    while True:
        try:
            sample = testDataiter.next()
        except StopIteration:
            break

        # sample is batch of images        
        # est_pose, opticalflow of a sample; qn how to get optical flow if it is just 1 image?
        motions, flow = testvo.test_batch(sample)
        # posenp and flownp

        avg_rmse, rmse_trans, rmse_rot = compute_frame_error(motions, sample['motion'])
        frames_avg_error.extend(avg_rmse)
        frames_trans_error.extend(rmse_trans)
        frames_rot_error.extend(rmse_rot)

        # add all the est_poses of a trajectory to list 
        motionlist.extend(motions)

        if args.save_flow:
            for k in range(flow.shape[0]):
                flowk = flow[k].transpose(1,2,0)
                np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                flow_vis = visflow(flowk)
                cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png',flow_vis)
                flowcount += 1

    # predicted poses from a traj
    poselist = ses2poses_quat(np.array(motionlist))

    # calculate ATE, RPE, KITTI-RPE, frames_error
    if args.pose_file.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        # eval one traj by one traj
        results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True, kittitype=(datastr=='kitti'))
        ate = results['ate_score']
        rpe = results['rpe_score']

        if datastr=='euroc':
            print("==> ATE: %.4f" %(results['ate_score']))
        else:
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname=args.results+testname+'.png', title='ATE %.4f' %(results['ate_score']))
        np.savetxt(args.results+testname+'.txt',results['est_aligned'])

        # print rpe
        print("==> RPE: %.4f, %.4f" %(rpe[0], rpe[1]))

        # save to file
        frames_exceeded_avg_thres = np.where(np.array(frames_avg_error) > args.threshold_error_avg)[0]
        frames_exceeded_trans_thres = np.where(np.array(frames_trans_error) > args.threshold_error_trans)[0]
        frames_exceeded_rot_thres = np.where(np.array(frames_rot_error) > args.threshold_error_rot)[0]
        save_to_file(args.results, args.scene, args.difficulty, args.traj_num, testDataset.__len__()+1, ate, rpe, args.threshold_error_avg, args.threshold_error_trans, args.threshold_error_rot, frames_avg_error, frames_exceeded_avg_thres, frames_trans_error, frames_exceeded_trans_thres, frames_rot_error, frames_exceeded_rot_thres)

    else:
        np.savetxt(args.results+testname+'.txt',poselist)
