import numpy as np

#########standard metrics from FCRN and OmniDepth
def standard_metrics(input_gt_depth_image,pred_depth_image, verbose=True):
    ##########################################################
    input_gt_depth = input_gt_depth_image.copy()
    pred_depth = pred_depth_image.copy()

    n = np.sum(input_gt_depth > 1e-3) ####valid pixels
                        
    ###CHECK mask - meters
    idxs = ( (input_gt_depth <= 1e-3) )
    pred_depth[idxs] = 1
    input_gt_depth[idxs] = 1

    print('valid samples:',n,'masked samples:', np.sum(idxs))

    ####STEP 1: compute delta################################################################
    #######prepare mask
    pred_d_gt = pred_depth / input_gt_depth
    pred_d_gt[idxs] = 100
    gt_d_pred = input_gt_depth / pred_depth
    gt_d_pred[idxs] = 100

    Threshold_1_25 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n
    Threshold_1_25_2 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
    Threshold_1_25_3 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n
    ########################################################################################        

    #####STEP 2: compute mean errors##########################################################
    #####CHECK for invalid pixels or max out of range
    input_gt_depth_norm = input_gt_depth / np.max(input_gt_depth)
    pred_depth_norm = pred_depth / np.max(pred_depth)
        
    log_pred = np.log(pred_depth_norm)
    log_gt = np.log(input_gt_depth_norm)
               
    ###OmniDepth: 
    RMSE_linear = ((pred_depth - input_gt_depth) ** 2).mean()
    RMSE_log = np.sqrt(((log_pred - log_gt) ** 2).mean())
    ARD = (np.abs((pred_depth_norm - input_gt_depth_norm)) / input_gt_depth_norm).mean()
    SRD = (((pred_depth_norm - input_gt_depth_norm)** 2) / input_gt_depth_norm).mean()
   

    if(verbose):
        print('Threshold_1_25: {}'.format(Threshold_1_25))
        print('Threshold_1_25_2: {}'.format(Threshold_1_25_2))
        print('Threshold_1_25_3: {}'.format(Threshold_1_25_3))
        print('RMSE_linear: {}'.format(RMSE_linear))
        print('RMSE_log: {}'.format(RMSE_log))
        print('SRD (MRE): {}'.format(SRD))
        print('ARD (MAE): {}'.format(ARD))
        
    return Threshold_1_25,Threshold_1_25_2,Threshold_1_25_3, RMSE_linear,RMSE_log,ARD,SRD


