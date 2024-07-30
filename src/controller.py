import gc
import os.path

import torch
import torch.nn.functional as F

from flow.flow_utils import flow_warp

# AdaIn


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class AttentionControl():

    def __init__(self, inner_strength, mask_period, cross_period, ada_period,
                 warp_period):
        self.step_store = self.get_empty_store()
        self.cur_step = 0
        self.total_step = 0
        self.cur_index = 0
        self.init_store = False
        self.restore = False
        self.update = False
        self.flow = None
        self.mask = None
        self.restorex0 = False
        self.updatex0 = False
        self.inner_strength = inner_strength
        self.cross_period = cross_period
        self.mask_period = mask_period
        self.ada_period = ada_period
        self.warp_period = warp_period

        self.frame = 0
        self.block_idx = 3

        self.attn = None
        self.cur_frame_x = None

        self.all_frame_inv_feature_path = "/home/yfyuan/YYF/Rerender/exp/attn_map/all_frames_inv_features"
        self.batch_frame_attn_feature_path = "/home/yfyuan/YYF/Rerender/exp/attn_map/batch_frame_attn_feature"

        self.threshold_block_idx = [3, 4]
        self.steps = [x for x in range(20)]
        self.cur_step = 0

        self.random_noise = False
        self.random_th = 9
        self.random_pos_1 = (0.0, 0.0)
        self.random_pos_2 = (0.0, 0.0)

    @staticmethod
    def get_empty_store():
        return {
            'first': [],
            'previous': [],
            'x0_previous': [],
            'first_ada': []
        }

    def forward(self, context, is_cross: bool, place_in_unet: str):
        # cross_period = (self.total_step * self.cross_period[0],
        #                 self.total_step * self.cross_period[1])
        self.attn = None
        self.cur_frame_x = None
        if not is_cross and place_in_unet == 'up':
            if self.init_store :
                self.step_store['first'].append(context.detach())
                self.step_store['previous'].append(context.detach())
                if self.block_idx in self.threshold_block_idx:
                    self.attn = torch.load(os.path.join(self.batch_frame_attn_feature_path,
                                                            f"attn_{self.block_idx}_frame_{0}.pt")).to(context.device)
            # if self.update:
            #     tmp = context.clone().detach()
            # if self.restore and self.cur_step >= cross_period[0] and \
            #         self.cur_step <= cross_period[1]:
            #
            #     if self.block_idx < 0:
            #         first_context = torch.load(os.path.join(self.all_frame_inv_feature_path,
            #                                                 f"sa_in_{self.block_idx}_frame_{0}.pt")).to(context.device)
            #         previous_context = torch.load(os.path.join(self.all_frame_inv_feature_path,
            #                                                 f"sa_in_{self.block_idx}_frame_{self.frame}.pt")).to(context.device)
            #         self.cur_frame_x = torch.load(os.path.join(self.all_frame_inv_feature_path,
            #                                                 f"sa_in_{self.block_idx}_frame_{self.frame+1}.pt")).to(context.device)
            #         context = torch.cat(
            #             (first_context,
            #              previous_context),
            #             dim=1).clone()
            #     else:
            #         context = torch.cat(
            #             (self.step_store['first'][self.cur_index],
            #              self.step_store['previous'][self.cur_index]),
            #             dim=1).clone()
            # if self.update:
            #     self.step_store['previous'][self.cur_index] = tmp
            self.cur_index += 1
            self.block_idx += 1
        return context

    def update_x0(self, x0):
        if self.init_store:
            self.step_store['x0_previous'].append(x0.detach())
            style_mean, style_std = calc_mean_std(x0.detach())
            self.step_store['first_ada'].append(style_mean.detach())
            self.step_store['first_ada'].append(style_std.detach())
        if self.updatex0:
            tmp = x0.clone().detach()
        if self.restorex0:
            if self.cur_step >= self.total_step * self.ada_period[
                    0] and self.cur_step <= self.total_step * self.ada_period[
                        1]:
                x0 = F.instance_norm(x0) * self.step_store['first_ada'][
                    2 * self.cur_step +
                    1] + self.step_store['first_ada'][2 * self.cur_step]
            if self.cur_step >= self.total_step * self.warp_period[
                    0] and self.cur_step <= self.total_step * self.warp_period[
                        1]:
                pre = self.step_store['x0_previous'][self.cur_step]
                x0 = flow_warp(pre, self.flow, mode='nearest') * self.mask + (
                    1 - self.mask) * x0
        if self.updatex0:
            self.step_store['x0_previous'][self.cur_step] = tmp
        return x0

    def set_warp(self, flow, mask):
        self.flow = flow.clone()
        self.mask = mask.clone()

    def __call__(self, context, is_cross: bool, place_in_unet: str):
        context = self.forward(context, is_cross, place_in_unet)
        return context

    def set_step(self, step):
        self.cur_step = step

    def set_total_step(self, total_step):
        self.total_step = total_step
        self.cur_index = 0

    def clear_store(self):
        del self.step_store
        torch.cuda.empty_cache()
        gc.collect()
        self.step_store = self.get_empty_store()

    def set_task(self, task, restore_step=1.0):
        self.init_store = False
        self.restore = False
        self.update = False
        self.cur_index = 0
        self.restore_step = restore_step
        self.updatex0 = False
        self.restorex0 = False
        if 'initfirst' in task:
            self.init_store = True
            self.clear_store()
        if 'updatestyle' in task:
            self.update = True
        if 'keepstyle' in task:
            self.restore = True
        if 'updatex0' in task:
            self.updatex0 = True
        if 'keepx0' in task:
            self.restorex0 = True
