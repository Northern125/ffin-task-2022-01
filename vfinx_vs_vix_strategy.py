import logging
from pandas import Series

from backtesting import PortfolioBacktest


def _calc_alloc(_vix, thresh_1=18, thresh_2=23, thresh_3=30):
    _alloc = 1

    if thresh_1 <= _vix < thresh_2:
        _alloc = .5
    elif thresh_2 <= _vix < thresh_3:
        _alloc = .25
    elif _vix >= thresh_3:
        _alloc = 0
    else:
        _alloc = 1

    return _alloc


class VIXCounteringStrategyBacktest(PortfolioBacktest):
    def run_strategy(self,
                     vix: Series,
                     opens: Series,
                     calc_alloc: callable = _calc_alloc,
                     *args,
                     **kwargs):
        logger = logging.getLogger(f'{self.logger.name}.run_strategy')

        vix = vix.copy()
        sec = self.securities[0]

        for i, date in enumerate(self.dates[:-1]):
            next_date = self.dates[i + 1]

            todays_cashflow = self.cashflows.loc[date].sum()
            if todays_cashflow != 0:
                self.positions.loc[date, 'cash'] += todays_cashflow
                self.calc_pos_nav_alloc(date)

            vix_curr = vix.loc[date]
            alloc_curr = self.allocation.loc[date, sec]

            alloc_calcd = calc_alloc(vix_curr, *args, **kwargs)
            logger.debug(f'date: {date}, next_date:{next_date}, alloc_calcd: {alloc_calcd}, alloc_curr: {alloc_curr}')

            if alloc_curr != alloc_calcd:
                alloc_change = alloc_calcd - alloc_curr
                value_change = alloc_change * self.nav.loc[date]
                position_change = value_change / self.quotes.loc[date, sec]
                positions_change = {sec: position_change}

                rebalance_price = opens.loc[next_date]
                rebalance_prices = {sec: rebalance_price}

                logger.debug(f'alloc_curr != alloc_calcd, rebalancing; '
                             f'date: {date}, next_date: {next_date}, '
                             f'alloc_change: {alloc_change}, '
                             f'value_change: {value_change}, '
                             f'position_change: {position_change}, '
                             f'rebalance_price: {rebalance_price}')

                self.do_rebalance(date,
                                  next_date,
                                  positions_change=positions_change,
                                  rebalance_prices=rebalance_prices)

            else:
                self.keep(date, next_date)
