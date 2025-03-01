import os
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from einops import rearrange
from utils import generate_click_prompt
from function import transform_prompt
import torchvision.utils as vutils
import cfg


def vis_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None, boxes = None):
    args = cfg.parse_args()
    
    b,c,h,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2: # for REFUGE multi mask output
        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    elif c > 2: # for multi-class segmentation > 2 classes
        preds = []
        gts = []
        for i in range(0, c):
            pred = pred_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            preds.append(pred)
            gt = gt_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            gts.append(gt)
        tup = [imgs[:row_num,:,:,:]] + preds + gts
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    else:
        imgs = torchvision.transforms.Resize((h,w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        if points != None:
            for i in range(b):
                if args.thd:
                    ps = np.round(points.cpu()/args.roi_size * args.out_size).to(dtype = torch.int)
                else:
                    ps = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                # gt_masks[i,:,points[i,0]-5:points[i,0]+5,points[i,1]-5:points[i,1]+5] = torch.Tensor([255, 0, 0]).to(dtype = torch.float32, device = torch.device('cuda:' + str(dev)))
                for p in ps:
                    gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
                    gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
                    gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4
        if boxes is not None:
            for i in range(b):
                # the next line causes: ValueError: Tensor uint8 expected, got torch.float32
                # imgs[i, :] = torchvision.utils.draw_bounding_boxes(imgs[i, :], boxes[i])
                # until TorchVision 0.19 is released (paired with Pytorch 2.4), apply this workaround:
                img255 = (imgs[i] * 255).byte()
                img255 = torchvision.utils.draw_bounding_boxes(img255, boxes[i].reshape(-1, 4), colors="red")
                img01 = img255 / 255
                # torchvision.utils.save_image(img01, save_path + "_boxes.png")
                imgs[i, :] = img01
        tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)

def predict_sam(args, pred_loader, epoch, net: torch.nn.Module, clean_dir=True):
    """
    SAM模型预测函数，只进行前向推理并保存预测mask的可视化结果。
    """
    net.eval()
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    # 确保保存预测结果的文件夹存在
    sample_path = args.path_helper['sample_path']
    os.makedirs(sample_path, exist_ok=True)

    with tqdm(total=len(pred_loader), desc='Predicting', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(pred_loader):
            # 获取输入图像
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            # 如果未提供点提示，则通过 generate_click_prompt 生成（此处 masks 参数可传 None）
            if 'pt' not in pack or args.thd:
                imgsw, ptw, _ = generate_click_prompt(imgsw, None)
                # 同时构造一个默认的 point_labels
                point_labels = torch.ones(ptw.shape[:1], device=GPUdevice)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label'].to(device=GPUdevice)
            # 获取文件名信息，用于保存时命名
            name = pack['image_meta_dict']['filename_or_obj']

            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))

            # 对长轴方向按块进行推理（若未设置 evl_chunk，则一次处理整张图）
            while (buoy + evl_ch) <= imgsw.size(-1):
                # 根据是否使用thd决定如何截取点提示
                if args.thd:
                    pt = ptw[:, :, buoy: buoy + evl_ch]
                else:
                    pt = ptw

                # 截取当前块的图像
                imgs = imgsw[..., buoy: buoy + evl_ch]
                buoy += evl_ch

                # 若使用 thd 预处理，则调整形状和尺寸
                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
                    imgs = imgs.repeat(1, 3, 1, 1)
                    # 这里默认所有样本都存在点提示
                    point_labels = torch.ones(imgs.size(0), device=GPUdevice)
                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)

                showp = pt

                # 若点提示不为 -1，则转换为网络所需的格式
                if point_labels.clone().flatten()[0] != -1:
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    if len(point_labels.shape) == 1:  # 单点提示
                        coords_torch, labels_torch, showp = coords_torch[None, :, :], labels_torch[None, :], showp[None, :, :]
                    pt = (coords_torch, labels_torch)

                imgs = imgs.to(dtype=torch.float32, device=GPUdevice)

                # 前向推理，获得预测 mask
                with torch.no_grad():
                    imge = net.image_encoder(imgs)
                    if args.net in ['sam', 'mobile_sam']:
                        se, de = net.prompt_encoder(
                            points=pt,
                            boxes=None,
                            masks=None,
                        )
                    elif args.net == "efficient_sam":
                        # 转换提示格式（需自行实现 transform_prompt）
                        coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, imgs.size(-2), imgs.size(-1))
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                    if args.net == 'sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=(args.multimask_output > 1),
                        )
                    elif args.net == 'mobile_sam':
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                        )
                    elif args.net == "efficient_sam":
                        se = se.view(se.shape[0], 1, se.shape[1], se.shape[2])
                        pred, _ = net.mask_decoder(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            multimask_output=False,
                        )

                    # 将预测结果 resize 到设定的输出尺寸
                    pred = F.interpolate(pred, size=(args.out_size, args.out_size))

                    # 构造保存文件名
                    namecat = 'Predict'
                    for na in name[:2]:
                        img_name = na.split('/')[-1].split('.')[0]
                        namecat += img_name + '+'
                    save_path = os.path.join(sample_path, namecat + 'epoch_' + str(epoch) + '.jpg')

                    # 调用可视化函数，将预测 mask 结果叠加到原图上并保存
                    # 注意：此处传入的第三个参数为 None，表示没有 ground truth mask 用于对比
                    vis_image(imgs, pred, None, save_path, reverse=False, points=showp)

                # end while

            pbar.update()
