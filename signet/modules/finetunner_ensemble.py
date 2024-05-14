import torch

from signet.models.finetuner import FineTuner

class FinetunnerEnsemble(FineTuner):
    def __init__(self, finetuners):
        super(FinetunnerEnsemble, self).__init__()
        self.finetuners = finetuners
        
    def forward(self,
                mutation_dist,
                baseline_guess,
                num_mut,
                cutoff=0.01):

        outputs = [
            finetuner(
                mutation_dist,
                baseline_guess,
                num_mut,
                cutoff)
            for finetuner in self.finetuners
            ]
        
        outputs = torch.stack(outputs, dim=0).mean(dim=0)
        outputs = self._apply_cutoff(outputs, cutoff=cutoff)
        return outputs