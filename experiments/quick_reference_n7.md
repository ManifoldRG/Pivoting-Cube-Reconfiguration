# Quick Reference: Best Training Commands for 7 Agents

## 🎯 Why n=7 is Different

At n=7, dimension reduction becomes **critical**:
- **Full matrix**: 56 dimensions (7² + 7) - Too many!
- **Four-band**: 35 dimensions (7×4 + 7) - **37.5% reduction** ✅
- **Four-band + local k=3**: 19 dimensions - **66% reduction** ✅✅

**Key insight**: At n=7, you NEED dimension reduction to scale effectively.

---

## 🚀 Recommended Starting Point

**Best approach for n=7 (with dimension reduction):**
```bash
python train/train_sb3.py \
  --num_agents 7 \
  --max_steps 1000 \
  --episodes 2000 \
  --use_four_band_reduction \
  --enable_soft_matching_reward \
  --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 \
  --success_bonus 100.0 \
  --lr 3e-4 \
  --gamma 0.99 \
  --lam 0.95 \
  --clip 0.2 \
  --n_steps 2048 \
  --batch_size 128 \
  --epochs 10 \
  --hidden_dim 256 \
  --entropy_coef 0.01
```

---

## 📊 Dimension Reduction Comparison

| Approach | Obs Dims | Reduction | Expected Performance | Scalability |
|----------|----------|-----------|---------------------|-------------|
| **Full Matrix** | 56 | - | May struggle | ❌ Poor (O(n²)) |
| **Four-Band** | 35 | 37.5% | Good baseline | ✅ Good (O(n)) |
| **Four-Band + k=3** | 19 | 66% | Slight drop? | ✅✅ Excellent (O(k)) |
| **Four-Band + k=5** | 27 | 52% | Near baseline | ✅ Very Good (O(k)) |
| **Four-Band + k=7** | 35 | 37.5% | Same as four-band | ✅ Good (O(k)) |

---

## 🧪 Individual Test Commands

### Critical Comparison: Full vs Four-Band

**Full Matrix (56 dims) - May struggle**
```bash
python train/train_sb3.py --num_agents 7 --max_steps 1000 --episodes 2000 \
  --enable_soft_matching_reward --soft_matching_scale 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

**Four-Band (35 dims) - Recommended**
```bash
python train/train_sb3.py --num_agents 7 --max_steps 1000 --episodes 2000 \
  --use_four_band_reduction \
  --enable_soft_matching_reward --soft_matching_scale 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

**Performance Gap = Impact of Dimension Reduction**

### Local Neighborhood Variants

**Four-Band + Local k=3 (19 dims) - Most aggressive**
```bash
python train/train_sb3.py --num_agents 7 --max_steps 1000 --episodes 2000 \
  --use_four_band_reduction --use_local_neighborhood --local_neighborhood_k 3 \
  --enable_soft_matching_reward --soft_matching_scale 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

**Four-Band + Local k=5 (27 dims) - Balanced**
```bash
python train/train_sb3.py --num_agents 7 --max_steps 1000 --episodes 2000 \
  --use_four_band_reduction --use_local_neighborhood --local_neighborhood_k 5 \
  --enable_soft_matching_reward --soft_matching_scale 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

### Network Capacity Experiments

**Large Network to Compensate for Reduction**
```bash
python train/train_sb3.py --num_agents 7 --max_steps 1000 --episodes 2000 \
  --use_four_band_reduction \
  --hidden_dim 512 \
  --enable_soft_matching_reward --soft_matching_scale 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

---

## 📈 Expected Results & Success Criteria

### Realistic Targets for n=7

| Configuration | Target Success Rate | Notes |
|---------------|-------------------|-------|
| Full Matrix (56d) | 30-50% | High dimensionality hurts |
| Four-Band (35d) | 60-80% | Should enable learning |
| Four-Band + Local k=3 (19d) | 50-70% | Trade-off: dims vs info |
| Four-Band + Local k=5 (27d) | 65-85% | Sweet spot? |
| Four-Band + Large Net | 70-85% | Capacity helps |

**Success = Four-band achieves >70% while full matrix struggles**

This proves dimension reduction **enables scaling**!

---

## 🔬 Quick Test Suite (6 tests, ~4-5 hours)

```bash
./experiments/quick_test_n7.sh
```

**Tests:**
1. Full Matrix (56d) - Baseline (likely struggles)
2. Four-Band (35d) - Primary approach
3. Four-Band + Local k=3 (19d) - Most aggressive
4. Four-Band + Local k=5 (27d) - Balanced
5. Four-Band + Large Network - Capacity test
6. Potential + Four-Band - Alternative reward

**Key Question**: Does dimension reduction enable learning at n=7?

---

## 📊 Comprehensive Suite (15 tests, ~18-24 hours)

```bash
./experiments/test_suite_n7.sh
```

**Sections:**
1. **Dimension Reduction** (5 tests) - Full, four-band, local k=3/5/7
2. **Network Capacity** (3 tests) - Small/medium/large networks
3. **Reward Comparison** (3 tests) - Soft matching, potential, hybrid
4. **Extended Training** (2 tests) - 4000 episodes for validation

---

## 🎯 Analysis Strategy

### Primary Question
**Does four-band reduction enable scaling from n=5 to n=7?**

### What to Compare

**n=5 Baseline** (from first test suite)
- Full matrix: ~85% success
- Four-band: ~82% success
- Gap: ~3% (acceptable)

**n=7 Target**
- Full matrix: May fail (<50%)
- Four-band: Should work (>70%)
- **Conclusion**: Reduction enables scaling! ✅

### Success Metrics

✅ **Excellent**: Four-band achieves >75% at n=7
✅ **Good**: Four-band achieves 60-75% at n=7
⚠️ **Acceptable**: Four-band achieves 50-60% at n=7
❌ **Poor**: Four-band <50% at n=7 (need to debug)

---

## 💡 Hyperparameter Tuning for n=7

### If Four-Band Performance is Poor

**Try larger network:**
```bash
--hidden_dim 512
```

**Try more exploration:**
```bash
--entropy_coef 0.02
```

**Try more training:**
```bash
--episodes 4000
```

**Try different reward scaling:**
```bash
--soft_matching_scale 150.0  # Increase reward signal
```

### If Local Neighborhood Performs Poorly

**Increase k:**
```bash
--local_neighborhood_k 5  # More context
```

**Use larger network:**
```bash
--hidden_dim 512  # Compensate for lost info
```

---

## 🔍 TensorBoard Analysis

### Compare Dimension Reduction Impact
```bash
tensorboard --logdir_spec=\
full:results/n7_<timestamp>/1.1_full_matrix,\
four:results/n7_<timestamp>/1.2_four_band,\
local:results/n7_<timestamp>/1.3_four_band_local_k3
```

**Look for:**
- Does full matrix show learning at all?
- How much faster does four-band converge?
- What's the final performance gap?

### Key Metrics
1. **custom/success_rate** - Most important!
2. **custom/avg_episode_length** - Efficiency
3. **reward/episode** - Learning signal
4. **loss/value_loss** - Value function quality

---

## 📋 Parallel Execution Strategy

### Run n=5 and n=7 in Parallel

**Terminal 1: n=5 baseline**
```bash
./experiments/quick_test_n5.sh
```

**Terminal 2: n=7 scaling test**
```bash
./experiments/quick_test_n7.sh
```

**Terminal 3: Monitor both**
```bash
tensorboard --logdir_spec=\
n5:results/quick_n5_<timestamp>,\
n7:results/quick_n7_<timestamp>
```

**Total time**: ~3 hours (both run in parallel)

---

## 🎓 Interpreting n=7 Results

### Scenario 1: Four-Band Enables Scaling ✅
```
Full Matrix (56d):     45% success  (struggles)
Four-Band (35d):       78% success  (works!)
Four-Band + k=3 (19d): 65% success  (slight drop)
```
**Conclusion**: Dimension reduction successfully enables scaling!
**Next**: Scale to n=10 with four-band + local

### Scenario 2: All Approaches Struggle ❌
```
Full Matrix (56d):     35% success
Four-Band (35d):       42% success
Four-Band + k=3 (19d): 38% success
```
**Conclusion**: n=7 is too hard with current approach
**Actions**:
- Increase max_steps to 1500
- Try larger network (512)
- Longer training (4000 episodes)
- Check if n=5 worked first!

### Scenario 3: Full Matrix Works Fine 🤔
```
Full Matrix (56d):     82% success  (unexpected!)
Four-Band (35d):       85% success
```
**Conclusion**: n=7 isn't actually hard, reduction not critical yet
**Next**: Need to test n=10+ to see where reduction becomes necessary

---

## ⚡ Quick Commands Cheat Sheet

```bash
# Quick test suite
./experiments/quick_test_n7.sh

# Full test suite
./experiments/test_suite_n7.sh

# Single best test (four-band)
python train/train_sb3.py --num_agents 7 --max_steps 1000 --episodes 2000 \
  --use_four_band_reduction --enable_soft_matching_reward --lr 3e-4

# Analyze results
python experiments/analyze_results.py results/n7_<timestamp>

# Compare n=5 vs n=7
python experiments/analyze_results.py results/n5_<timestamp> --section 2 > n5_reduction.txt
python experiments/analyze_results.py results/n7_<timestamp> --section 1 > n7_reduction.txt
diff n5_reduction.txt n7_reduction.txt
```

---

## 🎯 Decision Tree After n=7 Tests

```
RESULTS from n=7
  │
  ├─ Four-band >70% success?
  │   ├─ YES → Dimension reduction validated! ✅
  │   │   └─ Action: Scale to n=10 with four-band
  │   │
  │   └─ NO → Need investigation
  │       ├─ Is full matrix also poor (<60%)?
  │       │   ├─ YES → n=7 too hard, need better approach
  │       │   └─ NO → Four-band has issues, debug
  │       │
  │       └─ Try: Larger network, more training
  │
  └─ Local k=3 performance?
      ├─ >60% → Excellent! Can scale further ✅✅
      ├─ 50-60% → Acceptable trade-off ✅
      └─ <50% → Use k=5 or four-band only
```

---

## 🚀 Ready to Scale!

**Next steps after n=7:**
1. If four-band works → Test n=10
2. If local k=3 works → Test n=15
3. Write paper comparing n=5 vs n=7 vs n=10+

The dimension reduction should shine at n=7 and beyond! 🌟
