📈 Forecast-Driven Portfolio Optimization Engine

A production-ready financial portfolio allocation system that integrates ARIMA-based return forecasting, rolling backtesting, risk-aware optimization, and automated testing.

This project demonstrates both quantitative finance methodology and professional software engineering practices, aligning with Week 12 objectives for portfolio forecasting and performance evaluation.

🎯 Business Objective

Design and implement a robust portfolio allocation system that:

Forecasts asset returns using time-series modeling

Constructs optimized portfolios based on expected return and risk

Evaluates performance via rolling backtesting

Measures risk-adjusted performance using institutional-grade metrics

Ensures engineering reliability via testing and CI automation

The goal is to simulate a forecast-driven asset allocation workflow similar to what is used in quantitative investment strategies.

🏗 System Architecture
src/
│
├── data/                # Data fetching & preprocessing
├── models/              # ARIMA forecasting logic
├── portfolio/           # Optimization & performance metrics
├── backtest/            # Rolling backtest engine
└── main.py              # End-to-end pipeline

Engineering principles applied:

  Modular separation of concerns

  Deterministic pipeline execution

  Numerical stability safeguards

  Fallback allocation logic

  Automated testing (pytest)

  CI pipeline (GitHub Actions)

📊 Assets Used

  TSLA (Equity – Growth)

  SPY (Equity – Broad Market)

  BND (Bond ETF – Fixed Income)

These provide cross-asset diversification for realistic portfolio construction.

🔮 Forecasting Methodology
  Model: ARIMA

  Log-transformed prices

  Time-aware train/test split

  Fixed ARIMA order for rolling stability

  Efficient forecasting without repeated expensive auto-selection

Purpose:
  Estimate expected returns to drive forward-looking allocation decisions.

📐 Portfolio Construction
  Step 1: Expected Return Estimation

    Forecast-based expected returns.

  Step 2: Risk Estimation

    Covariance matrix of asset returns.

    Regularization applied to prevent singular matrix errors.

  Step 3: Optimization

    Mean-variance optimization with:

    Numerical stability controls

    Equal-weight fallback if optimization fails

🔁 Rolling Backtest

    A rolling window framework simulates real-world investment behavior:

    Forecast → Allocate → Observe return → Update portfolio

    Prevents look-ahead bias

    Evaluates strategy robustness

📉 Performance Metrics

The following institutional-grade metrics are implemented:

  Metric	Description
  CAGR	Compound Annual Growth Rate
  Annualized Volatility	Risk (standard deviation)
  Sharpe Ratio	Risk-adjusted return
  Max Drawdown	Worst peak-to-trough decline
  Sample Output
  CAGR: 0.0300
  Annualized Volatility: 0.0420
  Sharpe Ratio: 0.2495
  Max Drawdown: -0.0517

Interpretation:

  Stable low-volatility allocation

  Controlled drawdowns (~5%)

  Positive but modest risk-adjusted performance

🧪 Engineering & Reliability
Unit Testing

5 pytest tests covering:

  CAGR calculation

  Volatility correctness

  Sharpe ratio behavior

  Max drawdown logic

  Realistic data robustness

  All tests pass.

  Continuous Integration

  GitHub Actions workflow automatically:

  Installs dependencies

  Runs pytest

  Verifies stability on push

This ensures reproducibility and production-readiness.

▶️ How to Run

1️⃣ Create virtual environment

python -m venv .venv

2️⃣ Activate environment

Windows:

.venv\Scripts\activate

Mac/Linux:

source .venv/bin/activate

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run pipeline

python -m src.main

📌 Key Improvements Implemented (Week 12)

  Stable rolling ARIMA forecasting

  Covariance matrix regularization

  Fallback allocation logic

  Risk metrics integration

  Automated unit testing

  CI/CD pipeline

⚠️ Limitations

  ARIMA assumes linear dynamics

  Small rolling windows may limit statistical power

  No transaction cost modeling

  No regime-switching logic

  Future enhancements could include:

  GARCH volatility modeling

  Bayesian return forecasting

  Black-Litterman allocation

  Transaction cost modeling

  Monte Carlo simulation

📈 Strategic Insight

This system demonstrates how forecast-driven allocation can:

  Improve decision structure

  Quantify risk exposure

  Maintain disciplined portfolio construction

  Provide measurable performance evaluation

It bridges:
Quantitative finance + Robust software engineering.

📄 License

Educational and research use.

👨‍💻 Author

Portfolio Forecasting & Optimization – Week 12 Project
Built with Python, ARIMA, Mean-Variance Optimization, and CI automation.
