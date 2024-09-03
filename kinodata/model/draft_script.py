#%%
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl


#%%

class UncertaintyAwareLoss(nn.Module):

    def __init__(self, weight_pki=1, weight_pose=1):
        super(UncertaintyAwareLoss, self).__init__() #do I need this?
        #self.loss_pki=loss_pki #do I need this?
        #self.loss_pose=loss_pose #do I need this?
        self.weight_pki = weight_pki
        self.weight_pose = weight_pose

    def forward(self, pred, batch):

        print(batch)

        target_activity = batch.target_activity
        target_pose_certainty=batch.target_pose_certainty
        #activity_mask = batch.activity_mask

        pred = self.forward(batch, 3).view(-1, 1)
        print(pred)
        print(pred[:, 0])
        pred_activity = pred[:, 0]
        pred_unc_activity = pred[:, 1]
        pose_certainty = pred[:, 2]

       


        loss_activity = (((target_activity-pred_activity).pow(2) / pred_unc_activity.pow(2))*pose_certainty)#.mean() #why do I need the mean?

        loss_pose = target_pose_certainty*np.log(pose_certainty)+(1-target_pose_certainty)*np.log(1-pose_certainty)

        regulariser_term=1/pred_unc_activity #change this later?


        total_loss = self.weight_activity * loss_activity + self.weight_pose * loss_pose + regulariser_term

        # for logging return losses separately
        return loss_activity, loss_pose, total_loss

# %%
# Generate some dummy data
batch_size = 10
pred = torch.rand(batch_size, 3)  # Random predictions for (activity, uncertainty in activity, pose certainty)
batch = torch.rand(batch_size, 3)  # Random target activity values, target pose certainty values, and additional data

# Initialize the loss function
criterion = UncertaintyAwareLoss(weight_pki=1, weight_pose=1)

# Compute the loss
loss_activity, loss_pose, total_loss = criterion(pred, batch)

# Print the results
print(f"Loss Activity: {loss_activity.item()}")
print(f"Loss Pose: {loss_pose.item()}")
print(f"Total Loss: {total_loss.item()}")
# %%
kinodata = torch.load('/home/raquellrdc/Desktop/postdoc/fast_ml/kinodata-3D-affinity-prediction/kinodata/model/test_kinodata.pth')
david=torch.load('/home/raquellrdc/Desktop/postdoc/fast_ml/kinodata-3D-affinity-prediction/kinodata/model/test_david.pth')

# %%
kinodata.y
# %%
david.y
# %%
if torch.isnan(david.y):
    print('yes')
# %%
