# Round-3 models

`C/checkpoint.pt` is the BC candidate selected on the development arena before final evaluation. `D/checkpoint.pt` is the enhanced-feature alternative with stronger imitation accuracy; neither is proven to beat the official course baseline.

- A/B: tiny CNN, basic/enhanced features.
- C/D: small ResNet, basic/enhanced features.
- A0: intentionally frozen-backbone control; do not use as the production training default.
- PPO_C/ppo_best.pt: iteration-10 PPO negative-result artifact; it underperformed BC. The protected overall eval-best is byte-identical to C.
- mahjong_v2_C / mahjong_v2_D: source ZIP and standalone CPU deployment weights; locally tested, not uploaded to Botzone.
- references: historical in-project BC/RL checkpoints used as fixed controls/opponents, preserved byte-for-byte.

Checkpoints include Python/NumPy RNG and optimizer states and should only be loaded from trusted sources. Deployment `.pt` files contain state dictionaries. Identity, byte size, training configuration and provenance are in `reports/round3/ARTIFACT_MANIFEST.json`, `raw/selection.json`, and the detailed reports. Training used the public IJCAI corpus identified in those reports; its upstream README declares MIT. No raw game corpus is redistributed here.
