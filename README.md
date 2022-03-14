# portfolio-backtesting
Backtester for portfolio algo strategies. Given the logic of allocation, the module performs backtesting and calculates various metrics like total return, annualized return, excess return etc.

## How to use
To implement a custom strategy, one should inherit `PortfolioBacktest` class and re-define the `run_strategy` method
