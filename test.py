import pdb

import torch.nn as nn
import torch
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        assert cost_class != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs.shape[:2]

            # Final cost matrix
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            targets = torch.nn.functional.normalize(targets, p=2, dim=-1)
            cost_matrix = outputs @ targets.permute(0, 2, 1)
            C = self.cost_class * cost_matrix
            C = C.view(bs, num_queries, -1).cpu()

            # sizes = [num_queries]
            # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            # return [(torch.as_tensor(j, dtype=torch.int64)) for _, j in indices]

            indices = [[linear_sum_assignment(c)[0], linear_sum_assignment(c)[1]] for c in C]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


if __name__ == '__main__':
    loss = HungarianMatcher()
    in1 = torch.randn([4, 30, 768])
    in2 = torch.randn([4, 30, 768])
    out = loss(in1, in2)
    print(out[0].shape)
