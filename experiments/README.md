# MSSA Experiments: Test Suite for 5 Agents

This directory contains a comprehensive test suite for evaluating different training approaches for the MSSA project with 5 agents.

## 📁 Files Overview

| File | Description | Run Time |
|------|-------------|----------|
| `test_suite_n5.sh` | **Full test suite** - 18 comprehensive tests | ~12-16 hours |
| `quick_test_n5.sh` | **Quick validation** - 5 essential tests | ~2-3 hours |
| `analyze_results.py` | **Results analysis** - Compare test outcomes | Instant |
| `quick_reference_n5.md` | **Command reference** - All individual commands | N/A |

## 🚀 Quick Start

### Option 1: Run Quick Test Suite (Recommended First)
```bash
# Run 5 essential tests (~2-3 hours)
./experiments/quick_test_n5.sh
```

### Option 2: Run Full Test Suite
```bash
# Run all 18 tests (~12-16 hours)
./experiments/test_suite_n5.sh
```

### Option 3: Run Individual Test
```bash
# See quick_reference_n5.md for all commands
python train/train_sb3.py --num_agents 5 --max_steps 500 --episodes 2000 \
  --enable_soft_matching_reward --soft_matching_scale 100.0 \
  --lr 3e-4 --gamma 0.99 --n_steps 2048 --batch_size 128
```

## 📊 Test Suite Structure

### Quick Test Suite (5 tests)
1. **Soft Matching** - Doc.tex baseline
2. **Potential Reward** - Alternative baseline
3. **Soft + Four-Band** - Test dimension reduction
4. **Soft + Four-Band + Local** - Aggressive reduction
5. **Hybrid** - Combined rewards

### Full Test Suite (18 tests)

**Section 1: Reward Comparison (5 tests)**
- Test different reward functions
- Find best reward strategy
- Compare soft matching, potential, bounty, and hybrids

**Section 2: Dimension Reduction (4 tests)**
- Full matrix baseline
- Four-band reduction
- Four-band + local neighborhood (k=3)
- Four-band + local neighborhood (k=5)

**Section 3: Hyperparameter Ablation (5 tests)**
- Learning rate variations
- Network size
- Entropy coefficient
- Soft matching decay rate

**Section 4: Extended Training (4 tests)**
- Long runs (5000 episodes)
- Validate convergence
- Check final performance

## 📈 Analyzing Results

### Quick Check During Training
```bash
# Watch progress in real-time
tail -f results/n5_<timestamp>/1.1_soft_matching/training_sb3.log
```

### View in TensorBoard
```bash
# All results
tensorboard --logdir=results/n5_<timestamp>

# Specific test
tensorboard --logdir=results/n5_<timestamp>/1.1_soft_matching

# Compare two tests
tensorboard --logdir_spec=test1:results/n5_<timestamp>/1.1_soft_matching,test2:results/n5_<timestamp>/1.2_potential
```

### Generate Analysis Report
```bash
# Analyze all results
python experiments/analyze_results.py results/n5_<timestamp>

# Analyze specific section
python experiments/analyze_results.py results/n5_<timestamp> --section 1

# Save detailed report
python experiments/analyze_results.py results/n5_<timestamp> --output my_report.md
```

### Quick Success Rate Comparison
```bash
# Extract all success rates
grep -r "Success rate:" results/n5_<timestamp>/*/training_sb3.log
```

## 🎯 What to Look For

### Primary Success Metrics
1. **Final Success Rate** - Should be >80% for n=5
2. **Learning Curve** - Smooth increase over episodes
3. **Sample Efficiency** - Episodes needed to reach 50% success
4. **Convergence** - Stable performance in last 20% of training

### Warning Signs
- 🚩 Success rate <50% after 2000 episodes
- 🚩 Highly unstable learning (wild oscillations)
- 🚩 No improvement after first 500 episodes
- 🚩 Average episode length = max_steps (not reaching goal)

### Good Signs
- ✅ Steady improvement over training
- ✅ Success rate >80% in final episodes
- ✅ Decreasing episode length over time
- ✅ Stable policy in extended training

## 📋 Expected Outcomes

### Soft Matching Reward
- **Expected**: 80-95% success rate
- **Why**: Matches doc.tex theoretical framework
- **Best for**: Clean, principled approach

### Potential Reward
- **Expected**: 70-85% success rate
- **Why**: Simpler, may be less sample efficient
- **Best for**: Baseline comparison

### Dimension Reduction (n=5)
- **Four-band**: ~Same performance as full (dimensions similar)
- **Local (k=3)**: Possible 5-10% performance drop
- **Importance**: Validates approach before scaling to larger n

### Hyperparameters
- **Learning rate**: 3e-4 typically optimal
- **Network size**: 256 sufficient for n=5
- **Entropy**: 0.01 balances exploration/exploitation

## 🔬 Experimental Design Rationale

### Why These Tests?

**Section 1: Reward Comparison**
- Identifies best signal for agent learning
- Critical foundation for all other tests
- Validates doc.tex theoretical approach

**Section 2: Dimension Reduction**
- Tests if reduction maintains performance at n=5
- Prepares for scaling to n>5 where it's essential
- Validates observation representation choices

**Section 3: Hyperparameter Ablation**
- Fine-tunes learning for best results
- Identifies robust vs. sensitive parameters
- Helps debug if results are poor

**Section 4: Extended Training**
- Confirms convergence (not just luck)
- Checks for overfitting
- Validates final production config

## 🛠️ Troubleshooting

### Test Fails to Run
```bash
# Check Python environment
python --version  # Should be 3.8+

# Check dependencies
pip install stable-baselines3 sb3-contrib gymnasium tensorboard

# Check file structure
ls train/train_sb3.py  # Should exist
```

### Out of Memory
```bash
# Reduce batch size
--batch_size 64

# Reduce network size
--hidden_dim 128

# Reduce rollout steps
--n_steps 1024
```

### Training Too Slow
```bash
# Reduce episodes for testing
--episodes 500

# Use smaller network for quick test
--hidden_dim 128

# Check GPU usage
nvidia-smi  # If available
```

## 📚 Next Steps After Testing

### If Results Are Good (>80% success)
1. ✅ Scale up to n=7-10 agents
2. ✅ Use four-band reduction for larger n
3. ✅ Test on more complex configurations
4. ✅ Write paper with results

### If Results Are Poor (<60% success)
1. 🔍 Check environment is correct (visualize episodes)
2. 🔍 Verify reward signal is non-zero
3. 🔍 Try longer training (--episodes 5000)
4. 🔍 Adjust hyperparameters (higher entropy)

### If Dimension Reduction Hurts Performance
1. 🔍 Check observation reconstruction accuracy
2. 🔍 Try larger local neighborhood (k=5 for n=5)
3. 🔍 Increase network size to compensate
4. 🔍 May need architectural changes

## 📖 Additional Resources

- **Doc.tex** - Theoretical foundation for approaches
- **train/train_sb3.py --help** - All available CLI options
- **quick_reference_n5.md** - Individual command examples
- **TensorBoard docs** - https://www.tensorflow.org/tensorboard

## 🎓 Understanding the Results

### Success Rate Comparison (Section 1)
```
Test 1.1 (Soft Matching):  92.5%  ← Winner if highest
Test 1.2 (Potential):      85.3%
Test 1.3 (Bounty):         78.1%
Test 1.4 (Hybrid):         88.7%
Test 1.5 (All):            83.2%
```
**Conclusion**: Use soft matching for future tests

### Dimension Reduction Impact (Section 2)
```
Test 2.1 (Full):           92.5%  ← Baseline
Test 2.2 (Four-band):      91.8%  ← -0.7% (acceptable)
Test 2.3 (Four+Local k=3): 87.2%  ← -5.3% (trade-off)
Test 2.4 (Four+Local k=5): 92.1%  ← -0.4% (excellent)
```
**Conclusion**: Four-band safe, local k=3 has cost but enables scaling

## ⚙️ Customizing the Test Suite

### Add New Test
Edit `test_suite_n5.sh` and add:
```bash
echo "Running Test X.Y: My Custom Test..."
python train/train_sb3.py $BASE_ARGS \
  --your_custom_flags \
  --log_dir "$RESULTS_DIR/X.Y_my_test"
```

### Skip Tests
Comment out tests in the script:
```bash
# echo "Running Test 1.3: Bounty Reward..."
# python train/train_sb3.py ...
```

### Adjust Test Parameters
Edit `BASE_ARGS` in the script:
```bash
BASE_ARGS="
  --num_agents 5
  --max_steps 1000  # Increase max steps
  --episodes 500    # Reduce episodes for faster testing
  ...
"
```

---

**Ready to start testing!** Run `./experiments/quick_test_n5.sh` for a quick validation. 🚀
