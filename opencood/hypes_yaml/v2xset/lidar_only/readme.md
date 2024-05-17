# To train fpv-rcnn & fvoxel-rcnn

To train FPV-RCNN efficiently, it is recommended to first train first stage and then train the whole network.
An example training schedule could be: 

1. Train stage1 for 20 epochs: use fpvrcnn.yaml as reference, make sure that stage2 is inactive

```yaml
model:
  args:
    activate_stage2: False
```

2. Train stage1 and stage2 **for another 20 epochs**: set ```activate_stage2``` to ```True``` and ```epoches``` to ```40```, resume parameters from step 1 and train futher. 


Note that fvoxel-rcnn stage2 seems only accept batchsize to be 1.