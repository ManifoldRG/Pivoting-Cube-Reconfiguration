# Quick Reference: Best Training Commands for 5 Agents

## 🎯 Recommended Starting Point

**Best baseline (based on doc.tex):**
```bash
python train/train_sb3.py \
  --num_agents 5 \
  --max_steps 500 \
  --episodes 2000 \
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

## 📊 Test Matrix: What Each Test Does

| Test | Focus | Key Parameters | Expected Outcome |
|------|-------|----------------|------------------|
| **1.1** | Soft matching reward (doc) | `--enable_soft_matching_reward` | Baseline for comparison |
| **1.2** | Potential reward | `--enable_potential_reward` | Compare against soft matching |
| **1.3** | Bounty reward | `--enable_bounty_reward` | Alternative sparse reward |
| **1.4** | Hybrid (soft + potential) | Both enabled, scaled down | May improve sample efficiency |
| **1.5** | All rewards | All enabled | Check if complementary |
| **2.1** | Full matrix obs | No reduction flags | Baseline observation |
| **2.2** | Four-band reduction | `--use_four_band_reduction` | 30→25 dims, test scalability |
| **2.3** | Four-band + local (k=3) | `--use_local_neighborhood --local_neighborhood_k 3` | 30→19 dims, most aggressive |
| **2.4** | Four-band + local (k=5) | `--local_neighborhood_k 5` | 30→25 dims, same as full |
| **3.1** | Higher learning rate | `--lr 1e-3` | Faster but less stable? |
| **3.2** | Lower learning rate | `--lr 1e-4` | Slower but more stable? |
| **3.3** | Larger network | `--hidden_dim 512` | More capacity |
| **3.4** | More exploration | `--entropy_coef 0.05` | Better exploration? |
| **3.5** | Slower reward decay | `--soft_matching_decay_beta 0.9995` | Maintain reward signal longer |
| **4.1** | Extended training | `--episodes 5000` | Final performance check |
| **4.2** | Extended + reduction | Four-band + 5000 episodes | Scalability validation |

---

## 🔬 Individual Test Commands

### Section 1: Reward Comparison

**1.1 - Soft Matching (Primary)**
```bash
python train/train_sb3.py --num_agents 5 --max_steps 500 --episodes 2000 \
  --enable_soft_matching_reward --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 --success_bonus 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

**1.2 - Potential Reward (Baseline)**
```bash
python train/train_sb3.py --num_agents 5 --max_steps 500 --episodes 2000 \
  --enable_potential_reward --potential_scale 1.0 \
  --potential_normalize n2 --success_bonus 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

**1.3 - Bounty Reward**
```bash
python train/train_sb3.py --num_agents 5 --max_steps 500 --episodes 2000 \
  --enable_bounty_reward --bounty_gamma 0.999 --bounty_eta 2.0 \
  --bounty_total_frac_of_success 0.3 --success_bonus 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

### Section 2: Dimension Reduction

**2.1 - Full Matrix (30 dims)**
```bash
python train/train_sb3.py --num_agents 5 --max_steps 500 --episodes 2000 \
  --enable_soft_matching_reward --soft_matching_scale 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

**2.2 - Four-Band (25 dims)**
```bash
python train/train_sb3.py --num_agents 5 --max_steps 500 --episodes 2000 \
  --use_four_band_reduction \
  --enable_soft_matching_reward --soft_matching_scale 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

**2.3 - Four-Band + Local k=3 (19 dims)**
```bash
python train/train_sb3.py --num_agents 5 --max_steps 500 --episodes 2000 \
  --use_four_band_reduction --use_local_neighborhood --local_neighborhood_k 3 \
  --enable_soft_matching_reward --soft_matching_scale 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

---

## 📈 Expected Results & Metrics to Track

### Primary Metrics
1. **Success Rate**: % of episodes reaching goal (most important!)
2. **Average Episode Length**: Steps to completion
3. **Learning Curve**: Success rate over time
4. **Sample Efficiency**: Episodes needed to reach 50%/80% success

### Secondary Metrics
1. **Φ (Phi) Score**: Soft matching score progression
2. **Reward Components**: Breakdown of reward sources
3. **Action Mask Usage**: % of actions masked
4. **Training Time**: Wall-clock time per episode

### Comparison Strategy
```
Section 1: Find best reward function
  → Compare final success rates
  → Winner becomes baseline for Section 2

Section 2: Test dimension reduction impact
  → Does reduction hurt performance for n=5?
  → Prepare for scaling to larger n

Section 3: Optimize hyperparameters
  → Fine-tune best config from Sections 1+2
  → Balance speed vs. stability

Section 4: Validate with extended training
  → Confirm convergence
  → Check for overfitting
```

---

## ⚡ Quick Start Guide

### Run Single Test
```bash
# Make script executable
chmod +x experiments/test_suite_n5.sh

# Run specific test manually
python train/train_sb3.py --num_agents 5 --max_steps 500 --episodes 2000 \
  --enable_soft_matching_reward --soft_matching_decay_beta 0.999 \
  --soft_matching_scale 100.0 --lr 3e-4
```

### Run Full Test Suite
```bash
# Run all tests (will take several hours)
./experiments/test_suite_n5.sh
```

### Run Subset of Tests
```bash
# Edit test_suite_n5.sh and comment out sections you don't need
# For example, keep only Section 1 (reward comparison)
```

### Monitor Progress
```bash
# In another terminal
tensorboard --logdir=results/n5_<timestamp>

# Or check specific experiment
tensorboard --logdir=results/n5_<timestamp>/1.1_soft_matching
```

---

## 🎛️ Hyperparameter Tuning Guide

### If Training is Unstable
- ✅ Reduce learning rate: `--lr 1e-4`
- ✅ Increase batch size: `--batch_size 256`
- ✅ Reduce clip range: `--clip 0.1`

### If Learning is Too Slow
- ✅ Increase learning rate: `--lr 5e-4`
- ✅ More training epochs: `--epochs 15`
- ✅ Increase entropy: `--entropy_coef 0.02`

### If Not Exploring Enough
- ✅ Higher entropy: `--entropy_coef 0.05`
- ✅ Larger network: `--hidden_dim 512`
- ✅ More steps per rollout: `--n_steps 4096`

### If Overfitting
- ✅ Lower network size: `--hidden_dim 128`
- ✅ Higher entropy: `--entropy_coef 0.02`
- ✅ More episodes: `--episodes 5000`

---

## 🔍 Analysis After Running Tests

### View All Results in TensorBoard
```bash
tensorboard --logdir=results/n5_<timestamp>
```

### Compare Specific Tests
```bash
tensorboard --logdir_spec=\
test1:results/n5_<timestamp>/1.1_soft_matching,\
test2:results/n5_<timestamp>/1.2_potential
```

### Extract Final Success Rates
Look in the log files:
```bash
grep "Success rate" results/n5_*/*/training_sb3.log
```

### Expected Success Rates (Rough Estimates)
- **n=5 baseline**: 60-80% success rate
- **With good reward**: 80-95% success rate
- **Well-tuned**: >95% success rate

---

## 🎯 Decision Tree After Tests

```
START
  │
  ├─ Section 1: Which reward is best?
  │   └─ If Soft Matching wins → Use for all future tests
  │
  ├─ Section 2: Does reduction hurt performance?
  │   ├─ If NO → Safe to use for scaling to larger n
  │   └─ If YES → May need architecture changes
  │
  ├─ Section 3: Best hyperparameters?
  │   └─ Use optimal settings for production training
  │
  └─ Section 4: Converged performance?
      ├─ If >95% success → Ready to scale up
      └─ If <80% success → Debug environment/rewards
```
