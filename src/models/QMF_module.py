import numpy as np
import torch 

class History(object):
    def __init__(self, n_data, device):
        self.correctness = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.max_correctness = 1
        self.device = device

    # correctness update
    def correctness_update(self, data_idx, correctness, confidence):
        #probs = torch.nn.functional.softmax(output, dim=1)
        #confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()
        data_idx = [idx[0] for idx in data_idx]

        self.correctness[data_idx] += correctness.cpu().numpy()
        self.confidence[data_idx] = confidence.cpu().detach().numpy()

    # max correctness update
    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        #data_max = float(self.max_correctness)
        data_max = float(self.correctness.max())

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, data_idx1, data_idx2):
        data_idx1 = data_idx1.cpu().numpy()
        cum_correctness1 = self.correctness[data_idx1]
        cum_correctness2 = self.correctness[data_idx2]
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)
        # make target pair
        n_pair = len(data_idx1)
        if n_pair == 1:
            target1 = cum_correctness1
            target2 = cum_correctness2
        else:
            target1 = cum_correctness1[:n_pair]
            target2 = cum_correctness2[:n_pair]
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.from_numpy(target).float().to(self.device)
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().to(self.device)

        return target, margin


class QMF_module():
    def __init__(self, args):
        self.args = args
        gpu_ids = args.gpu_ids if isinstance(args.gpu_ids, list) else [args.gpu_ids]
        gpu_ids = [ int(ids) for ids in gpu_ids]
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def init_history(self, n_data):
        self.audio_history  = History(n_data, self.device)
        self.visual_history = History(n_data, self.device)
        if self.args.dataset in ['IEMOCAP']:
            self.text_history = History(n_data, self.device)