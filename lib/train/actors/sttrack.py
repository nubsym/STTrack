from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search


class STTRACKActor(BaseActor):
    """ Actor for training the STTRACK"""

    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        pred_bbox = self.forward_pass(data)  # (x1,y1,x2,y2)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        loss, status = self.compute_losses(pred_bbox, gt_bboxes)

        return loss, status

    def forward_pass(self, data):
        feat_dict_list = []
        hs_spatial_query = []
        temporal_query_list = []
        out_pred = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(self.net(img=NestedTensor(template_img_i, template_att_i), mode='backbone'))

        # process the search regions (t-th frame)
        for i in range(self.settings.num_search):
            search_img = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 128, 128)
            search_att = data['search_att'][i].view(-1, *data['search_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(self.net(img=NestedTensor(search_img, search_att), mode='backbone'))
            seq_dict = merge_template_search(feat_dict_list)
            feat_dict_list.pop(-1)
            if len(temporal_query_list) > 0:
                query_embed = temporal_query_list[-1].squeeze(0)  # spatial_query_n+1 = temporal_query_n
            else:
                query_embed = None
            spatial_query, enc_mem = self.net(seq_dict=seq_dict, query_embed=query_embed, mode="transformer")
            hs_spatial_query.append(spatial_query.squeeze(0).transpose(0, 1))  # (N,B,C)
            # tq_n = CA(tq_n,cat(tq_1,...,tq_n))
            temporal_query, _ = self.net(img=hs_spatial_query[-1], seq_dict={
                "feat": torch.cat([x for x in hs_spatial_query], dim=0)}, mode="update")
            temporal_query_list.append(temporal_query.unsqueeze(0).transpose(1, 2))  # (1,b,n,c)
            # weights.append(weight)
            pred_box = self.net(img=temporal_query_list[-1], seq_dict=enc_mem, mode="head")[0]["pred_boxes"]
            out_pred.append(pred_box)  # (x,y,x,y)

        return out_pred

    def compute_losses(self, pred_bbox, gt_bboxes, return_status=True):
        giou_loss = []
        iou_pred = []
        l1_loss = []
        # Get boxes
        if len(pred_bbox) != len(gt_bboxes):
            raise ValueError("Network outputs is NAN! Stop Training")
        gt_boxes_vec = [box_xywh_to_xyxy(gt_bboxes[i]).view(-1, 4).clamp(min=0.0, max=1.0) for i in
                        range(len(gt_bboxes))]  # (B,4) --> (B,1,4) --> (B,N,4)
        pred_bbox_vec = [pred_bbox[i].view(-1, 4) for i in range(len(pred_bbox))]
        # compute weight loss
        for i in range(len(pred_bbox_vec)):
            try:
                giou_loss.append(self.objective['giou'](pred_bbox_vec[i], gt_boxes_vec[i])[0])  #
                iou_pred.append(self.objective['giou'](pred_bbox_vec[i], gt_boxes_vec[i])[-1])  #
            except:
                giou_loss.append(torch.tensor(0.0).cuda())
                iou_pred.append(torch.tensor(0.0).cuda())
        # compute l1 loss
            l1_loss.append(self.objective['l1'](pred_bbox_vec[i], gt_boxes_vec[i]))  # # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['l1'] * sum(giou_loss) + self.loss_weight['giou'] * sum(l1_loss)
        if return_status:
            # status for log
            mean_iou = (sum(iou_pred) / len(iou_pred)).detach()  # .mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": sum(giou_loss).item(),
                      "Loss/l1": sum(l1_loss).item(),
                      "IoU": mean_iou.item()
                      }
            return loss, status
        else:
            return loss
