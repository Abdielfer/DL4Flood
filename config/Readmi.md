Here we summarize some supported (tested) models, losses, metrics and optimizers configurations.
You can simply copy-paste the section into the coniguration/parameters 
##  Model
model:
  _target_: model_set.models.UNetFlood
  classes : 1
  in_channels : 1
  dropout: True   # (bool) Use dropout or not
  prob: 0.1  # float: 0. - 0.99

## Loss ###
loss_fn:
  _partial_: True 
  _target_: torch.nn.functional.binary_cross_entropy_with_logits #scr.losses.lovasz_hinge 

## Metric ##
metric_fn :
  _partial_: True 
  _target_: scr.losses.iou_binary 

## Optimizer ###
optimizer: # ADAM
  _target_: torch.optim.Adam
  maximize : False
  lr: 0.00001
  weight_decay : 0.9
 