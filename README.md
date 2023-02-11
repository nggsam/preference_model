Trains and compares a variety of preference models (reward models) with different losses and datasets.

### TODOs
- [x] Add model.
- [x] Add training code.
- [x] Add evaluation code.
- [x] Test complete workflow with 10% train and 10% eval data for one epoch.
- [x] Add requirements.txt.
- [x] Train to make sure that loss is going down.
- [x] Add metrics to measure accuracy while training.
- [ ] Try different configs:
  - [ ] Freeze some of the layers to avoid overfitting.
  - [ ] Train first layer for 0.1 epoch. Then train the other layers.
- [ ] Deepspeed with a config file.
- [x] Add Deepspeed config and try Deepspeed training.
- [ ] Try PyTorch compile.
- [ ] Compare different losses.
- [ ] Compare different datasets.
- [ ] Add synthetic datasets.
 
### Resources
- Code forked from https://github.com/CarperAI/trlx/tree/main/examples/summarize_rlhf.