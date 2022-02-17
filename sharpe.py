def calc_sharpe(returns_mean, returns_std, multiply_by=None):
    sharpe = returns_mean / returns_std

    if multiply_by is None:
        return sharpe
    else:
        return sharpe * multiply_by
