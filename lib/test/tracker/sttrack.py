import os

# for debug
import cv2
import numpy as np
import torch

from lib.models.sttrack import build_sttrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.sttrack_utils import Preprocessor
from lib.train.data.processing_utils import sample_target
from lib.utils.box_ops import clip_box, box_xyxy_to_cxcywh
from lib.utils.merge import merge_template_search


class STARK_S(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_S, self).__init__(params)
        network = build_sttrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = '/home/luohui/SYM/tracking/code/temporal/tracking/debug'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = 1  # params.save_all_boxes
        self.z_dict1 = {}
        # self.oz_dict1 = {}
        self.feat = []
        # self.feat1 = []
        self.hs_bbox = []
        self.temporal_query = None

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template)
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.temporal_query = None
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = [self.z_dict1, x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            query_embed = self.temporal_query
            # run the transformer
            spatial_query, enc_mem = self.network.forward_transformer(seq_dict=seq_dict, query_embed=query_embed)
            # self.feat.append(spatial_query)  # (b,n,c)
            # TIM
            self.feat.append(spatial_query.squeeze(0).transpose(0, 1))  # (b,n,c)
            if len(self.feat) > 1:
                hs_spatial_query = torch.cat([x for x in self.feat[:-1]], dim=0)
            else:
                hs_spatial_query = torch.cat([x for x in self.feat], dim=0)
            temporal_query, weight = self.network.update_head(self.feat[-1], hs_spatial_query)
            # temporal as the spatial query of next frame
            self.temporal_query = temporal_query
            # self.temporal_query = self.feat[-1].squeeze(0)
            out_dict_pred, _, weight_feat = self.network.forward_head(temporal_query.unsqueeze(0).transpose(1, 2),
                                                                      enc_mem, run_box_head=True)

            out_dict = box_xyxy_to_cxcywh(out_dict_pred['pred_boxes'])

            pred_boxes = out_dict.view(-1, 4)  # ['pred_boxes']
        #     pred_boxes = 0
        #     self.hs_bbox.append(pred_boxe)
        #     if len(self.hs_bbox) <= 1:
        #         pred_boxes = pred_boxe
        #     else:
        #         for i in range(len(weight.view(1, -1).mean(0))):
        #             pred_boxes += weight.view(1, -1).mean(0)[i] * self.hs_bbox[i]
        #     # print('weight:', weight.view(1, -1).mean(0))
        #     if len(weight) == 5:
        #         self.hs_bbox.pop(0)
        #         self.feat.pop(0)
        #         self.hs_bbox.pop(0)
        # pred_boxes[-2:] = pred_boxe[-2:]
        # self.hs_bbox[-1] = pred_boxes
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=10)
            # cv2.putText(image_BGR, '#{}'.format(self.frame_id), (1620, 110), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 3)
            x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)
            # heatmap = cv2.applyColorMap(
            #     cv2.resize(np.uint8(weight_feat * 255), (W, H)), cv2.COLORMAP_JET)
            heatmap = cv2.applyColorMap(
                cv2.resize(np.uint8(weight_feat * 255), (self.params.search_size, self.params.search_size)), cv2.COLORMAP_JET)
            image_cam = x_patch_arr + heatmap * 0.5
            # cv2.putText(image_cam, '#{}'.format(self.frame_id), (1620, 110), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 3)
            # cv2.putText(image_cam, '#{}'.format(self.frame_id), (10, 22), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            # save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            # cv2.imwrite(save_path, image_cam)
            save_path_ori = os.path.join(self.save_dir, "%04d_ori.jpg" % self.frame_id)
            cv2.imwrite(save_path_ori, x_patch_arr)

        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return STARK_S
