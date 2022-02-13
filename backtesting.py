from pandas import DataFrame
import logging


def perform_backtest(portfolio, risk_free, risk_free_annualized=True, annualization_factor=360):
    logger = logging.getLogger(f'{__name__}.perform_backtest')

    portfolio = portfolio.rename('value').copy()
    risk_free = risk_free.rename('risk free rate').copy()
    logger.debug(risk_free)

    portfolio = DataFrame(portfolio).copy()
    portfolio['daily return'] = portfolio['value'].pct_change(periods=1)
    portfolio['daily return ann'] = (portfolio['daily return'] + 1) ** annualization_factor - 1
    portfolio['total return'] = (portfolio['daily return'] + 1).cumprod() - 1

    portfolio = portfolio.merge(risk_free, how='left', left_index=True, right_index=True).copy()
    if risk_free_annualized:
        portfolio['risk free rate daily'] = (portfolio['risk free rate'] + 1) ** (1 / annualization_factor) - 1
        portfolio['excess return'] = portfolio['daily return ann'] - portfolio['risk free rate']
        portfolio['excess return daily'] = portfolio['daily return'] - portfolio['risk free rate daily']
    else:
        portfolio['excess return'] = portfolio['daily return'] - portfolio['risk free rate']

    portfolio['daily PnL'] = portfolio['value'].diff(periods=1)
    portfolio['total PnL'] = portfolio['daily PnL'].cumsum()

    return portfolio
