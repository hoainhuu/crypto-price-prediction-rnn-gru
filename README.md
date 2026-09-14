# Deep Learning for Cryptocurrency Price Forecasting: An Empirical Study on RNN vs. GRU

A comparative empirical study evaluating the performance, convergence, and generalization capabilities of Recurrent Neural Networks (RNN) and Gated Recurrent Units (GRU) architectures. This project conducts multi-asset time-series forecasting across three major cryptocurrencies: Bitcoin (BTC), Binance Coin (BNB), and Ethereum (ETH).

---

## Project Motivation & Objectives

Financial time-series data in cryptocurrency markets exhibits high volatility, non-linear dependencies, and sudden trend shifts. Standard feedforward networks fail to capture temporal context, whereas Recurrent Architectures process dynamic sequential inputs.

**Core Objectives:**
1. **Model Architecture Comparison:** Benchmark traditional RNN against GRU to evaluate how gating mechanisms handle vanishing gradients in multi-step predictions.
2. **Data-Split Sensitivity Analysis:** Test model stability and out-of-sample generalization across three distinct Train/Validation/Test split strategies: `8:1:1`, `7:1.5:1.5`, and `6:2:2`.
3. **Cross-Asset Evaluation:** Validate model robustness across assets with varying volatility profiles (BTC, BNB, ETH).

---

## Experimental Setup & Methodology

### 1. Data Preprocessing & Feature Engineering
- **MinMax Normalization:** Scaled target price values into the range [0, 1] to stabilize backpropagation gradients and ensure faster loss convergence.
- **Sliding Window Generation:** Structured historical sequential data into supervised look-back time windows $(t-N, \dots, t-1) \rightarrow t$.

### 2. Data Splitting Strategies
To evaluate model robustness against over-fitting, models were systematically trained and tested under three data distribution schemes:
- **80 : 10 : 10 Split:** Prioritizes historical depth (80% training data) to evaluate maximum learning capacity.
- **70 : 15 : 15 Split:** Standard balanced cross-validation strategy.
- **60 : 20 : 20 Split:** Evaluated on a larger unseen test set (20%) to measure real-world temporal generalization.

---

## Repository Structure

```text
crypto-price-prediction-rnn-gru/
├── BITCOIN/
│   ├── BITCOIN hình/               # Loss curves & actual vs. predicted price plots
│   ├── GRU-8_1_1.ipynb             # GRU on 80:10:10 split
│   ├── GRU-7_1.5_1.5.ipynb         # GRU on 70:15:15 split
│   ├── GRU-6_2_2.ipynb             # GRU on 60:20:20 split
│   ├── RNN-8_1_1.ipynb             # RNN on 80:10:10 split
│   ├── RNN-7_1.5_1.5.ipynb         # RNN on 70:15:15 split
│   └── RNN-6_2_2.ipynb             # RNN on 60:20:20 split
├── BNB/
│   ├── BNB hình/                   # Output visualization plots for BNB
│   └── [Notebooks for RNN & GRU across 3 data splits]
├── Ethereum/
│   ├── Etherum hình/               # Output visualization plots for ETH
│   └── [Notebooks for RNN & GRU across 3 data splits]
└── README.md                       # Main project documentation
