Untrained ProtoNet baseline accuracy is ~43% (vs ~20% chance) on 5-way episodes, confirmed genuine (not a bug) via label-shuffling control test — shuffled accuracy drops to ~21%, matching chance. Likely explained by informative random CNN features (architectural prior, not learning) — see Saxe et al. 2011.

## ProtoNet learning verification (Week 9)
- Before training: 34.95% val acc (100-episode avg)
- After 1500 episodes: 51.01% val acc (100-episode avg)
- Improvement: +16.07 pp, clears noise band (untrained std ~10%)
- Confirms training pipeline is correctly learning, not just noise
- Not yet a final benchmark number — original paper reports ~65-68%
  after much longer training (thousands of episodes)

  ## MAML gradient flow verification (Step 7)
- Tested that outer-loop loss backpropagates through all `inner_steps`
  inner-loop adaptations into the original meta-parameters
- Result: 0/18 parameters with missing gradients, all nonzero grad norms
- Confirms functional_call + create_graph=True correctly implements
  full second-order MAML (not a silently-broken first-order-only path)
- Note: conv bias grad norms are near-zero (~1e-5 to 1e-6) due to
  BatchNorm immediately following each conv layer, making bias
  largely redundant -- expected architectural interaction, not a bug