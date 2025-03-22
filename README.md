# Comparison between different Portfolio Optimization Stragegies

**Author : Han Chu**

This project is designed to visualize the Mean-variance portfolio optimization and compare performances of different strategies. The data are stock prices of 5 real Chinese listed companies, I downloaded from Wind Database, period is from 2019 to 2023. Inside the Mean-variance strategy, I simulated 1000 possible random portfolios to draw scatters, then the efficient frontier and the Global Minimum Variance, the Maximum Sharpe Ratio (the Optimal portfolio). In the app, you will see the correlation matrix, interactive slidebars which allow different returns and volatilities, tables and diagrams of Returns and Volatilities of different portfolio strategies. After applying strategies, you can download the Excel file to check the weight distribution of the portfolios you chose.

## Main Results
<img width="1057" alt="截屏2025-03-22 22 14 23" src="https://github.com/user-attachments/assets/1d9b11c6-f6da-479f-b430-99d4d7fbb4bd" />
<img width="1188" alt="截屏2025-03-22 22 14 41" src="https://github.com/user-attachments/assets/6471b604-e403-4b38-9233-3e49bfc5722a" />

# Theories

## Equal Weight and Market value weighted porfolio


First we start with Equal Weight and Market value weighted porfolio

---
**Equal Weight Portfolio**

An Equal Weight Portfolio assigns the same weight to each asset regardless of its market value. For a portfolio with $ n $ assets, each asset's weight is:

$$
w_i = \frac{1}{n}, \quad \text{for } i = 1, 2, \dots, n.
$$

The portfolio return then is:

$$
E(R_{\text{Equal}}) = \frac{1}{n}\sum_{i=1}^{n} E(r_i)
$$

and the portfolio risk (variance) is computed as:

$$
\sigma_{\text{Equal}}^2 = \frac{1}{n^2}\left( \sum_{i=1}^{n} \sigma_i^2 + \sum_{i=1}^{n}\sum_{\substack{j=1 \\ j \neq i}}^{n} \mathrm{Cov}_{ij} \right).
$$

Taking the square root gives the portfolio's standard deviation:

$$
\sigma_{\text{Equal}} = \sqrt{ \frac{1}{n^2}\left( \sum_{i=1}^{n} \sigma_i^2 + \sum_{i=1}^{n}\sum_{\substack{j=1 \\ j \neq i}}^{n} \mathrm{Cov}_{ij} \right) }.
$$


*Note:* The calculation of the portfolio risk for an equal weight portfolio follows the same variance formula as above with $ w_i = \frac{1}{n} $.

Also, very interesting fact that **Harry Markowitz**, in an interview late in life about his pension portfolio, responded, "**I thought, 'You know, if the stock market goes way up and I'm not in, I'll feel stupid. And if it goes way down and I'm in it, I'll feel stupid.' So I went 50-50.**" So he applied this **Equal weight stratege!**
(URL: https://www.stockopedia.com/content/value-investing-fact-and-fiction-does-cheap-beat-expensive-97647/)

---
**Market Value Weighted Portfolio**

In a Market Value Weighted Portfolio, each asset's weight is proportional to its market capitalization. Suppose $ M_i $ denotes the market value (capitalization) of asset $ i $. The weight $ w_i $ is given by:

$$
w_i = \frac{M_i}{\sum_{j=1}^{n} M_j}
$$

Thus, the portfolio return is calculated as:

$$
E(R_{MV}) = \sum_{i=1}^{n} w_i\,E(r_i)
$$

and its risk (standard deviation) is computed using the standard portfolio variance formula:

$$
\sigma_{MV} = \sqrt{ \sum_{i=1}^{n} w_i^2\,\sigma_i^2 + \sum_{i=1}^{n} \sum_{j=i+1}^{n} w_i\,w_j\,\mathrm{Cov}_{ij} }.
$$

---

## Maximum Variance Portfolio

In the context of portfolio optimization, the **Maximum Variance Portfolio** is the portfolio that has the highest possible variance (risk) among all feasible portfolios. Given a vector of asset weights $w = (w_1, w_2, \dots, w_n)^T$ and the covariance matrix of asset returns $ Sigma$, the portfolio variance is defined as:

$$
\sigma_p^2 = w^T \Sigma w
$$

The Maximum Variance Portfolio is obtained by solving the following optimization problem:

$$
\begin{aligned}
\max_{w} \quad & w^T \Sigma w \\
\text{subject to} \quad & \sum_{i=1}^{n} w_i = 1.
\end{aligned}
$$

This formulation finds the portfolio weights that maximize the portfolio’s variance, subject to the full-investment constraint. Although such a portfolio is rarely practical due to its extreme risk, it serves as a theoretical benchmark in portfolio analysis.

---

**Minimum Variance Portfolio:**

$$
\begin{aligned}
\min_{\{\omega_i\}} \quad & \mathrm{Var}(r_p) \\
\text{subject to} \quad & \bar{r}_p = \sum_{i=1}^n \omega_i \, E(r_i), \\
& \sum_{i=1}^n \omega_i = 1.
\end{aligned}
$$

---

## Mean-variance

**The Mean-Variance (Markowitz Modern Portfolio) Theory**

In the context of portfolio selection, Maximum Variance Theory involves choosing the portfolio from the feasible set that has the largest variance (risk). While Markowitz’s classic approach usually focuses on minimizing variance or maximizing the Sharpe ratio, one can, in theory, reverse the problem to pick the portfolio with the highest variance—though this is rarely practical in real-world investing. A maximum variance portfolio typically concentrates extreme weights in the riskiest assets, resulting in very high volatility.

The Markowitz approach to portfolio optimization is commonly referred to as Mean-Variance (MV) optimization. It is based on the mean (expected return) and variance(risk) of assets within a portfolio.

**The portfolio’s return**
In Markowitz’s framework, the portfolio’s return is calculated as a weighted sum of the returns of the individual assets. That is,

$$
r_p = \sum_{i=1}^{n} w_i\,r_i
$$

where:

- $ r_p $ is the portfolio’s return,
- $ w_i $ is the weight of the $ i $-th asset in the portfolio,
- $ r_i $ is the return of the $ i $-th asset,
- $ n $ is the total number of assets.
  
If we consider the expected returns, the portfolio’s expected return is given by:

$$
E(r_p) = \sum_{i=1}^{n} w_i\,E(r_i)
$$

where:

- $ E(r_p) $ is the expected return of the portfolio,
- $ E(r_i) $ is the expected return of the $ i $-th asset.

---

**The portfolio's risk** (i.e., the standard deviation) is given by:

$$
\sigma_{\text{port}}
= \sqrt{
    \sum_{i=1}^n w_i^2 \,\sigma_i^2 
    \;+\; \sum_{i=1}^n \sum_{j=i+1}^n w_i \, w_j \,\mathrm{Cov}_{ij}
}
$$

where:

- $ \sigma_{\text{port}} $ = the standard deviation of the portfolio  
- $ w_i $ = the weight of the $i$-th asset in the portfolio  
- $ \sigma_i^2 $ = the variance of rates of return for asset $ i $  
- $ \mathrm{Cov}_{ij} $ = the covariance between rates of return for assets $ i $ and $ j $

![image.png](attachment:66a64640-d54a-4989-a761-0d858ebb39c8.png)
**Capital allocation line**
$$
\mathrm{CAL}: \; E(r_C) = r_F + \sigma_C \frac{E(r_P) - r_F}{\sigma_P}
$$


If investors can purchase a risk free asset with some return $ r_F $, then all correctly priced risky assets or portfolios will have expected return of the form

$$ E(R_P) = r_F + b\,\sigma_P $$

where $\ b $ is some incremental return to offset the risk (sometimes known as a risk premium), and $\sigma_P $ is the risk itself expressed as the standard deviation. By rearranging, we can see the risk premium has the following value:

$$ b = \frac{E(R_P) - r_F}{\sigma_P} $$

Substituting in our derivation for the risk premium above:

$$
E(R_C) = r_F + \sigma_C \frac{E(R_P) - r_F}{\sigma_P}
$$

This yields the Capital Allocation Line.

**Efficient frontier**

![image.png](attachment:2736627e-8195-484a-a0ac-5d52c467f884.png)

When CAL is tangent to Efficient Frontier, that point is the optimal portfolio, we can know the return and volatility of that portfolio, and the weights of each asset.
### **Tangency (Optimal) Portfolio**

When the Capital Allocation Line (CAL) is tangent to the Efficient Frontier, the tangency portfolio is achieved. This portfolio is optimal in the sense that it offers the highest risk-adjusted return (maximum Sharpe Ratio). At the tangency point, the slope of the CAL represents the Sharpe ratio of the tangency portfolio.

The tangency portfolio weights can be derived from the following formula:

$$
w^* = \frac{\Sigma^{-1}\left(E(r) - r_F\,\mathbf{I}\right)}{\mathbf{I}^T\,\Sigma^{-1}\left(E(r) - r_F\,\mathbf{I}\right)}
$$

Where:
- $ \Sigma $ is the covariance matrix of asset returns,
- $ E(r) $ is the vector of expected asset returns,
- $ r_F $ is the risk-free rate,
- $ \mathbf{I} $ is a vector of ones.

Once the optimal weights $ w^* $ are obtained, the tangency portfolio’s expected return and volatility are given by:

**Expected Return:**
$$
E(r_T) = w^{*T}E(r)
$$
**Volatility:**
$$
\sigma_T = \sqrt{w^{*T}\,\Sigma\,w^*}
$$

The maximum Sharpe ratio (slope of the CAL) is:

$$
\text{Sharpe Ratio} = \frac{E(r_T) - r_F}{\sigma_T}
$$

This tangency portfolio represents the best possible combination of risky assets when a risk-free asset is available. Any portfolio formed as a combination of the risk-free asset and the tangency portfolio will lie on the CAL, offering the optimal risk-return trade-off.

---

**Optimization Problem:**
$$
\max_w \quad w^T \mu - \frac{\gamma}{2} w^T \Sigma w
$$

Where:
- $ w $ = vector of portfolio weights,
- $ \mu $ = vector of expected asset returns,
- $ \Sigma $ = covariance matrix of asset returns,
- $ \gamma $ = risk aversion parameter.

This is the mean-variance optimization problem, where the goal is to maximize the expected return of the portfolio $ w^T \mu $ while penalizing for risk $ \frac{1}{2} w^T \Sigma w $.

**Optimal Weights Calculation:**
$$
w = \frac{1}{\gamma} \Sigma^{-1} \mu
$$

Where:
- $ w $ = optimal portfolio weights,
- $ \Sigma^{-1} $ = inverse of the covariance matrix,
- $ \mu $ = vector of expected returns,
- $ \gamma $ = risk aversion parameter.

The portfolio weights are obtained by solving for the optimal balance between return and risk using the covariance matrix and the expected returns. The risk aversion parameter $ \gamma $ determines how much risk the investor is willing to take in exchange for higher returns.

# Reference

Victor DeMiguel, Lorenzo Garlappi, Raman Uppal, Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?, The Review of Financial Studies, Volume 22, Issue 5, May 2009, Pages 1915–1953, https://doi.org/10.1093/rfs/hhm075


https://sites.math.washington.edu/~burke/crs/408/fin-proj/mark1.pdf

http://en.wikipedia.org/wiki/Capital_allocation_line

https://en.wikipedia.org/wiki/Efficient_frontier

The metrics I learned from this online course
Portfolio management with Python : EDHEC Buesness school URL: https://www.coursera.org/specializations/investment-management-python-machine-learning#courses

and their GitHub storage: 
URL: https://github.com/PeterSchuld/EDHEC_Investment-Management-with-Python-and-Machine-Learning-

mainly from MOOC1
URL: https://github.com/PeterSchuld/EDHEC_Investment-Management-with-Python-and-Machine-Learning-/blob/main/MOOC1_Introduction%20to%20Portfolio%20Construction/edhec_risk_kit_104.py

URL: https://github.com/PeterSchuld/EDHEC_Investment-Management-with-Python-and-Machine-Learning-/blob/main/MOOC1_Introduction%20to%20Portfolio%20Construction/edhec_risk_kit_105.py

Also I refered to other authors' work:
The general navigation page : 

URL: https://github.com/topics/portfolio-optimization

URL: https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/2-Mean-Variance-Optimisation.ipynb

Min-variance: https://zhuanlan.zhihu.com/p/658168978
