import torch

class AGM_module():
    def __init__(self, args):
        self.args = args
        self.iteration = 0
        self.train_score_a = 0.0
        self.train_score_v = 0.0
        self.train_score_t = 0.0
        
    def updata_train_score(self, score_audio, score_visual, score_text = None):
        self.iteration += 1
        
        score_audio = score_audio.item() if isinstance(score_audio, torch.Tensor) else score_audio
        score_visual = score_visual.item() if isinstance(score_visual, torch.Tensor) else score_visual
        
        self.train_score_a = self.train_score_a * (self.iteration - 1) / self.iteration  + score_audio / self.iteration 
        self.train_score_v = self.train_score_v * (self.iteration - 1) / self.iteration  + score_visual / self.iteration 
        
        if score_text is not None:
            score_text = score_text.item() if isinstance(score_text, torch.Tensor) else score_text
            self.train_score_t = self.train_score_t * (self.iteration - 1) / self.iteration  + score_text / self.iteration 