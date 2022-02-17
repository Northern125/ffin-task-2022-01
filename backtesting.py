from typing import Union

from pandas import DataFrame, Series, DatetimeIndex, concat, Timestamp
from numpy import nan
import logging


def perform_backtest(portfolio: Series, risk_free: Series, risk_free_annualized=True, annualization_factor=360):
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


def backtest_strategy(portfolio: DataFrame):
    portfolio = portfolio.copy()
    portfolio['rebalance'].replace(nan, 0, inplace=True)

    secs = portfolio.index.get_level_values('sec').unique().values
    sec = secs[0]

    if not (portfolio.groupby('date')['weight'].sum() == 1).all():
        raise Exception('Weights sum should be equal to 1 for each date')

    if (portfolio.xs('cash', axis='index', level='sec')['price'] != 1).any():
        raise Exception('Price of cash should always be 1')
    if (portfolio.xs('cash', axis='index', level='sec')['rebalance price'] != 1).any():
        raise Exception('Rebalance price of cash should always be 1')


class PortfolioBacktest:
    """
    Daily backtesting is assumed
    """
    def __init__(self,
                 securities: list,
                 dates: DatetimeIndex,
                 quotes: DataFrame,
                 capital: float = 1e6,
                 ini_positions: Series = None,
                 max_loan: float = 0):
        self.__name__ = 'PortfolioBacktesting'

        # logger
        self.logger = logging.getLogger(self.__name__)
        self.logger.info(f'Creating an instance of {self.logger.name}')

        # loan
        self.max_loan = max_loan

        # cols of each dataframe (securities list + cash)
        self.securities = securities
        _cols = securities + ['cash']

        # dates
        self.dates = dates

        # quotes (prices)
        self.quotes = quotes
        self.quotes['cash'] = 1
        self.logger.debug(f'quotes:\n{quotes}')

        # rebalance prices
        self.rebalance_prices = DataFrame(columns=_cols, index=dates, dtype=float)
        self.rebalance_prices['cash'] = 1
        self.logger.debug(f'rebalance_prices:\n{self.rebalance_prices}')

        # positions & positions values
        self.positions = DataFrame(columns=_cols, index=dates, dtype=float)
        if ini_positions is None:
            self.positions.iloc[0].loc['cash'] = capital
            _all_cols_but_cash = self.positions.columns[self.positions.columns != 'cash']
            self.positions.iloc[0].loc[_all_cols_but_cash] = 0
        else:
            self.positions.iloc[0] = ini_positions
        self.logger.debug(f'positions:\n{self.positions}')

        self.positions_values = (self.positions * self.quotes).copy()
        self.logger.debug(f'positions_values:\n{self.positions_values}')

        # NAV
        self.nav = Series(index=dates, dtype=float, name='NAV')
        self.nav.iloc[0] = capital
        self.logger.debug(f'nav:\n{self.nav}')

        # alloc
        self.allocation = DataFrame(columns=_cols, index=dates, dtype=float)
        self.allocation.iloc[0] = self.positions_values.iloc[0] / self.nav.iloc[0]
        self.logger.debug(f'allocation:\n{self.allocation}')

        # combined
        # _combined = {'price': self.quotes,
        #              'weight': self.allocation,
        #              '#': self.positions,
        #              'value': self.positions_values,
        #              'rebalance price': self.rebalance_prices}
        # self.portfolio = concat(_combined, axis='columns', join='outer').copy()
        # self.portfolio.columns.names = ['attr', 'sec']
        # self.portfolio['nav'] = self.nav
        # self.portfolio.iloc[1:] = nan
        #
        # self.logger.debug(f'portfolio (combined df):{self.portfolio}')

        # strategy
        self.strategy = None

    # def set_strategy(self, strategy: Strategy):
    #     self.strategy = strategy

    def _perform_sanity_check_date(self, date: Timestamp):
        allocation_check = self.allocation.loc[date].sum() == 1
        loan_check = self.positions_values.loc[date, 'cash'] >= - self.max_loan
        no_shorts_check = (self.positions.loc[date, self.securities] > 0).all()

        return allocation_check and loan_check and no_shorts_check

    def do_rebalance(self, date: Timestamp, next_date: Timestamp,
                     positions_change: dict, rebalance_prices: Union[dict, Series]):
        self.logger.info(f'Starting rebalance procedure. Date: {date}, next date: {next_date}, '
                         f'positions_change: {positions_change}, rebalance_prices:\n{rebalance_prices}')

        rebalance_prices = Series(rebalance_prices).copy()
        positions_change = Series(positions_change).copy()
        value_change = (rebalance_prices * positions_change).copy()

        # dealing w/ cash change
        positions_change.loc['cash'] = - value_change.sum()
        rebalance_prices.loc['cash'] = 1
        value_change.loc['cash'] = positions_change.loc['cash'] * rebalance_prices.loc['cash']

        self.logger.debug(f"""positions_change.loc['cash']: {positions_change.loc['cash']}""")
        self.logger.debug(f'value_change:\n{value_change}')

        # calculating next day positions, values, nav, alloc
        positions_next = self.positions.loc[date] + positions_change
        positions_values_next = self._calc_positions_values(positions_next,
                                                            self.quotes.loc[next_date])
        nav_next = self._calc_nav(positions_values_next)
        allocation_next = self._calc_allocation(positions_values_next, nav_next)

        self.logger.debug(f'positions_next:\n{positions_next}')
        self.logger.debug(f'positions_values_next:\n{positions_values_next}')
        self.logger.debug(f'nav_next: {nav_next}')
        self.logger.debug(f'allocation_next:\n{allocation_next}')

        # checking if rebalance can be done
        max_loan_check = positions_values_next.loc['cash'] > - self.max_loan
        no_shorts_check = (positions_next.loc[self.securities] > 0).all()
        rebalance_is_possible = max_loan_check and no_shorts_check
        self.logger.info(f'Rebalance is possible? - {rebalance_is_possible}')

        # performing rebalance if it is possible
        if rebalance_is_possible:
            self.rebalance_prices.loc[next_date] = rebalance_prices
            self.positions.loc[next_date] = positions_next
            self.positions_values.loc[next_date] = positions_values_next
            self.nav.loc[next_date] = nav_next
            self.allocation.loc[next_date] = allocation_next

            self.logger.info('Rebalance successfully completed')
        else:
            self.keep(date, next_date)
            self.logger.info(f'Rebalance can\'t be completed, this is why: '
                             f'max_loan_check: {max_loan_check}, '
                             f'no_shorts_check: {no_shorts_check}')

    def keep(self, date: Timestamp, next_date: Timestamp):
        self.positions.loc[next_date] = self.positions.loc[date]
        self.calc_positions_values(next_date)
        self.calc_nav(next_date)
        self.calc_allocation(next_date)

    @staticmethod
    def _calc_positions_values(positions: Series, quotes: Series):
        return (positions * quotes).copy()

    def calc_positions_values(self, date: Timestamp):
        self.positions_values.loc[date] = self._calc_positions_values(self.positions.loc[date], self.quotes.loc[date])

    @staticmethod
    def _calc_nav(positions_values: Series):
        return positions_values.sum()

    def calc_nav(self, date: Timestamp):
        self.nav.loc[date] = self._calc_nav(self.positions_values.loc[date])

    @staticmethod
    def _calc_allocation(positions_values: Series, nav: float):
        return (positions_values / nav).copy()

    def calc_allocation(self, date):
        self.allocation.loc[date] = self._calc_allocation(self.positions_values.loc[date], self.nav.loc[date])

    def apply_strategy(self, exogenous: Series):
        exogenous = exogenous.copy()
        sec = self.securities[0]

        # def _calc_alloc(_vix, thresh_1=18, thresh_2=23, thresh_3=30):
        #     _alloc = 1
        #
        #     if thresh_1 <= _vix < thresh_2:
        #         _alloc = .5
        #     elif _vix >= thresh_2:
        #         _alloc = .25
        #     elif _vix >= thresh_3:
        #         _alloc = 0
        #
        #     return _alloc

        for i, date in enumerate(self.dates[:-1]):
            next_date = self.dates[i + 1]
            # vix = exogenous.loc[date, 'vix']
            flag = exogenous.loc[date]
            alloc_curr = self.allocation.loc[date, sec]

            if flag == 1:
                positions_change = {sec: -1}
                rebalance_prices = self.quotes.loc[next_date]
                self.do_rebalance(date, next_date, positions_change, rebalance_prices)
            else:
                self.keep(date, next_date)

            # alloc_calcd = _calc_alloc(vix)

    def run(self, *args, **kwargs):
        self.apply_strategy(*args, *kwargs)

    def get_combined_df(self):
        _combined = {'price': self.quotes,
                     'weight': self.allocation,
                     '#': self.positions,
                     'value': self.positions_values,
                     'rebalance price': self.rebalance_prices}

        portfolio = concat(_combined, axis='columns', join='outer').copy()
        portfolio.columns.names = ['attr', 'sec']
        portfolio['nav'] = self.nav

        return portfolio
