# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

# 年化收益率 annualized return
def annualizedRet(r, year_days):
    '''
    @param r        : 每日收益率序列
    @param year_days: 一年中交易股票所有交易日期天数（股票开市的天数）一般是250天
    @return   : 年化收益率   
    '''

    cum_return = (1 + r).prod()
    total_days = r.shape[0]
    Annual_Returns = cum_return ** (year_days / total_days) - 1
    return Annual_Returns

# 年化波动率 annualized volatility
def annualizedVol(r, year_days, downside = False):
    '''
    @param r        : 每日收益率序列
    @param year_days: 一年中交易股票所有交易日期天数（股票开市的天数）一般是250天
    @param downside : 是否计算下行偏差
    @return: 年化波动率
    '''

    if downside:
        semistd = r[r < 0].std()
        return semistd * (year_days ** 0.5)
    else:
        return r.std() * (year_days ** 0.5)

# 最大回撤 maximum drawdown
def drawdown(r):
    '''
    @param r : 每日收益率序列
    @return: 最大回撤
    '''

    index = 1000 * (1 + r).cumprod()
    highwatermark = index.cummax()   # .cummax是找到当前节点之前的最大收益率
    drawdowns = (index - highwatermark) / highwatermark     # 计算回撤
    maxdrawdown = drawdowns.min()
    return maxdrawdown

# 偏度 skewness
def skewness(r):
    '''
    @param r : 每日收益率序列
    @return: 偏度
    '''
    centerMoment = r - r.mean()
    sigR = r.std(ddof=0)
    exp = (centerMoment ** 3).mean()
    return exp / sigR ** 3

# 峰度 kurtosis
def kurtosis(r):
    '''
    @param r : 每日收益率序列
    @return : 峰度
    '''
    centerMoment = r - r.mean()
    sigR = r.std(ddof=0)
    exp = (centerMoment ** 4).mean()
    return exp / sigR ** 4

# VaR
def varGaussian(r, level=5, modified=False):
    '''
    @param r : 每日收益率序列
    @param level: 置信水平
    @param modified: 使用泰勒展开式修正VaR
    @return: Value at Risk-风险价值
    '''
    from scipy.stats import norm
    z = norm.ppf(level / 100)

    if modified is True:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return - (r.mean() + z * r.std(ddof=0))

# 夏普比率 sharpe ratio (the slope)
def sharpeRatio(r, rf, year_days):
    '''
    @param r : 每日收益率序列
    @param rf : 无风险收益率 
    @param year_days : 一年当中有多少个交易日
    @return: 夏普比率，也称风险调整后的return
    '''
    # 将无风险收益率转换成日收益率，然后才能进行相加或者相减
    rf = (1 + rf) ** (1 / year_days) - 1
    excessRets = r - rf
    annExcessRets = annualizedRet(excessRets, year_days)
    annVol = annualizedVol(r, year_days)
    return annExcessRets / annVol

def sortinoRatio(r,rf, year_days):
    '''
    @param r : 每日收益率序列
    @param rf : 无风险收益率 
    @param year_days : 一年当中有多少个交易日
    @return : 夏普比率，也称风险调整后的return
    '''

    rf = (1 + rf) ** (1 / year_days) - 1
    excessRets = r - rf
    annExcessRets = annualizedRet(excessRets, year_days)
    anndownsideVol = annualizedVol(r, year_days, downside=True)
    return annExcessRets / anndownsideVol

def summary_stats(r, riskFree=0, periodsInYear=252):
    '''
    @param r : 每日收益率序列
    @param riskFree: 无风险收益率 
    @param year_days : 一年当中有多少个交易日
    @return: 各个策略的性能（DataFrame）
    '''

    if not isinstance(r,pd.DataFrame):
        r = pd.DataFrame(r)
    
    # aggregate表明每一列均调用此函数，聚集函数
    annR = r.aggregate(annualizedRet,year_days = periodsInYear)
    annVol = r.aggregate(annualizedVol, year_days= periodsInYear)
    dd = r.aggregate(lambda r: drawdown(r))
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    modVar = r.aggregate(varGaussian, level=5, modified=True)
    sharpe = r.aggregate(sharpeRatio, rf=riskFree, year_days = periodsInYear)
    sortino = r.aggregate(sortinoRatio, rf = riskFree, year_days = periodsInYear)

    stats = pd.DataFrame({
        'Annualized Returns': annR*100,
        'Annualized Volatility':  annVol*100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown':  dd*100,
        'Skewness': skew,
        'Kurtosis': kurt,
        'Cornish Fisher adj. VAR 5%': modVar*100,
    })

    #formatting
    stats['Annualized Returns'] = stats['Annualized Returns'].map('{:,.2f}%'.format)
    stats['Annualized Volatility'] = stats['Annualized Volatility'].map('{:,.2f}%'.format)
    stats['Sharpe Ratio'] = stats['Sharpe Ratio'].map('{:,.2f}'.format)
    stats['Sortino Ratio'] = stats['Sortino Ratio'].map('{:,.2f}'.format)
    stats['Max Drawdown'] = stats['Max Drawdown'].map('{:,.2f}%'.format)
    stats['Skewness'] = stats['Skewness'].map('{:,.2f}'.format)
    stats['Kurtosis'] = stats['Kurtosis'].map('{:,.2f}'.format)
    stats['Cornish Fisher adj. VAR 5%'] = stats['Cornish Fisher adj. VAR 5%'].map('{:,.2f}%'.format)

    return stats.T