import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datas import app_dir, data  # data is here
import seaborn as sns
from shiny import App, Inputs, Outputs, Session, render, ui, reactive
import os
#shiny run --reload --port=1145 --launch-browser ./test.py
'''
Portfolio optimization essentially belongs to the category of quantitative finance, which is an interdisciplinary field. It mainly relies on methods in econometrics and statistics, such as using historical data to estimate the returns and covariance of assets, and constructing mean-variance models to find the optimal combination. At the same time, machine learning methods have also been widely used in portfolio management in recent years, including return prediction, risk management, feature extraction, and dynamic strategy adjustment. Therefore, portfolio optimization is indeed closely related to econometrics, statistics, and machine learning.
'''
# Global configuration

risk_free = 0.02        # Approximately 10-year Chinese government bond yield 0.02
initial_amount = 10000  # Initial investment
code_list = ["600897", "300750", "601857", "300015", "601088"]
for code in code_list:
    print(code)


# 1. Data Preprocessing & Annualized Covariance
daily_return = pd.pivot_table(data, index="date", columns="code", values="return")
daily_return = daily_return / 100.0
market_size = pd.pivot_table(data, index="date", columns="code", values="marketcap")
market_size = market_size.bfill()
daily_return.index = pd.to_datetime(daily_return.index)

# Split into training set (for estimating returns/covariance) and test set (for validation/actual portfolio returns)
train = daily_return[daily_return.index < "2020-01-01"]
test_daily_return = daily_return[daily_return.index >= "2020-01-01"]

market_size.index = pd.to_datetime(market_size.index)
market_size = market_size[market_size.index >= "2020-01-01"]

stock_return = train
cov_mat = stock_return.cov()
cov_mat_annual = cov_mat * 252

# 2. Market Cap Weighted & Equal Weighted

def normalize_row(row):
    return row / row.sum()

market_size_norm = market_size.apply(normalize_row, axis=1)
mv_weight = test_daily_return.mul(market_size_norm, axis=1).sum(axis=1)

StockReturns = pd.DataFrame(mv_weight, columns=["marketcap weighted"])
StockReturns["equal weight"] = test_daily_return.mean(axis=1).values

# 3. Random Portfolios & Extracting Max Variance

n = len(code_list)
number = 1000 # Number of random portfolios
random_p = np.empty((number, n + 2))  # [weights..., Returns, Volatility]
np.random.seed(123)

for i in range(number):
    w_rand = np.random.random(n)
    w_rand /= w_rand.sum()   # No short selling
    # Estimate portfolio daily return on training data
    port_ret_daily = stock_return.mul(w_rand, axis=1).sum(axis=1).mean()
    annual_ret = (1 + port_ret_daily)**252 - 1
    annual_vol = np.sqrt(np.dot(w_rand.T, np.dot(cov_mat_annual, w_rand)))
    random_p[i, :n] = w_rand
    random_p[i, n] = annual_ret
    random_p[i, n+1] = annual_vol

RandomPortfolios = pd.DataFrame(
    random_p,
    columns=[code + "_weight" for code in code_list] + ["Returns", "Volatility"]
)

# Only take portfolios with positive returns
positive_returns = RandomPortfolios[RandomPortfolios["Returns"] > 0]
if not positive_returns.empty:
   # From random portfolios, find the one with minimum volatility (for reference)
    min_positive_index = positive_returns["Volatility"].idxmin()
    # From random portfolios, find the one with maximum volatility => Max Variance
    max_positive_index = positive_returns["Volatility"].idxmax()
    MaxVar_weights = np.array(RandomPortfolios.iloc[max_positive_index, 0:n])
else:
    # If no portfolio with positive return, set equal weights
    MaxVar_weights = np.full(n, 1.0/n)

MaxVar_weights_df = pd.DataFrame({"Code": code_list, "Weight": MaxVar_weights})
# calculate the daily return of this portfolio on test dataset 在测试集上计算该组合的每日收益
StockReturns["Max Variance"] = test_daily_return.mul(MaxVar_weights, axis=1).sum(axis=1)

# 4. Quadratic Programming: GMV, MSR, and Efficient Frontier Calculation

def global_min_var_portfolio(Sigma, short=False):
    """
    No short selling => w_i >= 0
    Objective: minimize w^T Sigma w
    Constraint: sum(w) = 1
    """
    n_ = Sigma.shape[0]
    def objective(w):
        return w @ Sigma @ w
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if short:
        bounds = None   # Allow short selling
    else:
        bounds = [(0, None)] * n_
    w0 = np.repeat(1/n_, n_)
    res = minimize(objective, w0, method="SLSQP", constraints=cons, bounds=bounds)
    if not res.success:
        raise ValueError("GMV optimization failed: " + res.message)
    return res.x

def msr_portfolio(Sigma, mu, risk_free=0.02, short=False):
    """
    Maximum Sharpe Ratio portfolio (Tangency portfolio)
    Minimize 0.5 * w^T Sigma w
    Subject to: (mu - rf)^T w = 1, and w_i >= 0 (if short=False)
    Finally, normalize to sum(w)=1
    """
    n_ = len(mu)
    excess = mu - risk_free
    def objective(w):
        return 0.5 * w @ Sigma @ w
    cons = [{"type": "eq", "fun": lambda w: w @ excess - 1}]
    if short:
        bounds = None
    else:
        bounds = [(0, None)] * n_
    w0 = np.repeat(1/n_, n_)
    res = minimize(objective, w0, method="SLSQP", constraints=cons, bounds=bounds)
    if not res.success:
        raise ValueError("MSR optimization failed: " + res.message)
    w_raw = res.x
    w_norm = w_raw / np.sum(w_raw)
    return w_norm

def min_var_given_return(mu, Sigma, target_return, short=False):
    """
    Given a target return, find the minimum variance portfolio
    Constraints: sum(w)=1, mu^T w = target_return, and w_i >= 0 (if short=False)
    """
    n_ = len(mu)
    def objective(w):
        return w @ Sigma @ w
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ mu - target_return}
    ]
    if short:
        bounds = None
    else:
        bounds = [(0, None)] * n_
    w0 = np.repeat(1/n_, n_)
    res = minimize(objective, w0, method="SLSQP", constraints=cons, bounds=bounds)
    if not res.success:
        return None, None, None
    w_opt = res.x
    var_ = w_opt @ Sigma @ w_opt
    ret_ = w_opt @ mu
    return w_opt, np.sqrt(var_), ret_

def compute_efficient_frontier(mu, Sigma, short=False, points=50):
    """
    Uniformly take several target return points in the interval [min(mu), max(mu)]
    and find the minimum variance portfolio for each, forming a continuous frontier.
    Returns a DataFrame: [Volatility, Returns, Weights]
    """
    min_ret = mu.min()
    max_ret = mu.max()
    frontier_data = []
    for target in np.linspace(min_ret, max_ret, points):
        w_opt, vol_, ret_ = min_var_given_return(mu, Sigma, target, short=short)
        if w_opt is not None:
            frontier_data.append((vol_, ret_, w_opt))
    frontier_data.sort(key=lambda x: x[0])
    df_front = pd.DataFrame(frontier_data, columns=["Volatility", "Returns", "Weights"])
    return df_front

# 5. Calculate GMV and MSR and store in StockReturns
mu_train = stock_return.mean() * 252  # Annual return estimate from training 
mu_np = mu_train.values
Sigma_np = cov_mat_annual.values

# GMV
gmv_weights = global_min_var_portfolio(Sigma_np, short=False)
GMV_weights_df = pd.DataFrame({"Code": code_list, "Weight": gmv_weights})
StockReturns["GMV"] = test_daily_return.mul(gmv_weights, axis=1).sum(axis=1)

# MSR
msr_weights = msr_portfolio(Sigma_np, mu_np, risk_free=risk_free, short=False)
MSR_weights_df = pd.DataFrame({"Code": code_list, "Weight": msr_weights})
StockReturns["MSR"] = test_daily_return.mul(msr_weights, axis=1).sum(axis=1)

# 6. Prepare Cumulative Returns
def get_cum_ret(daily_ret_series):
    total_amount = initial_amount
    acc = [total_amount]
    for r in daily_ret_series:
        total_amount = total_amount * r + total_amount
        acc.append(total_amount)
    return acc

# Extract required series
cum_ret_marketcap = get_cum_ret(StockReturns["marketcap weighted"])
cum_ret_equal = get_cum_ret(StockReturns["equal weight"])
cum_ret_gmv = get_cum_ret(StockReturns["GMV"])
cum_ret_msr = get_cum_ret(StockReturns["MSR"])

# 7. Shiny App: UI
app_ui = ui.page_fluid(
    ui.panel_title("Investment Portfolio Visualization"),
    
    # 1) Annualized Covariance Matrix Heatmap
    ui.card(
        ui.card_header("Annualized Covariance Matrix Heatmap"),
        ui.output_plot("heatmap_plot"),
        full_screen=True
    ),
    
    # 2) Custom Target Portfolio
    ui.card(
        ui.card_header("Custom Target Portfolio"),
        ui.input_slider("target_return", "Desired Return",
                        min=positive_returns["Returns"].min() if not positive_returns.empty else 0.0,
                        max=positive_returns["Returns"].max() if not positive_returns.empty else 0.1,
                        value=0.05, step=0.01),
        ui.input_slider("target_volatility", "Desired Volatility",
                        min=positive_returns["Volatility"].min() if not positive_returns.empty else 0.0,
                        max=positive_returns["Volatility"].max() if not positive_returns.empty else 0.1,
                        value=0.02, step=0.01),
        ui.layout_columns(
            ui.column(
                6,
                ui.card(
                    ui.card_header("Target Portfolio Chart"),
                    ui.output_plot("target_scatter_plot"),
                    full_screen=True
                )
            ),
            ui.column(
                8,
                ui.card(
                    ui.card_header("Target Portfolio Weights"),
                    ui.output_table("target_weight_table"),
                    full_screen=True
                )
            ),
            col_widths=[7, 8]
        )
    ),
    
    # 3) Strategy Selection (GMV, Max Variance, MSR)
    ui.card(
        ui.card_header("Strategy Selection Mean-Variance"),
        ui.input_checkbox_group(
            "portfolio_strategy", "Select Strategy",
            choices=["GMV", "Max Variance", "MSR"],
            selected=["GMV"]
        ),
        ui.input_action_button("apply_button", "Apply Strategy"),
        ui.download_button("download_weights", "Download Weights", filename="Strategy_Weights.xlsx"),
        ui.layout_columns(
            ui.column(
                8,
                ui.card(
                    ui.card_header("Strategy Weights Table"),
                    ui.output_table("strategy_table"),
                    full_screen=True
                )
            ),
            ui.column(
                8,
                ui.card(
                    ui.card_header("Portfolios (GMV / Max Variance / MSR) with Efficient Frontier and CAL"),
                    ui.output_plot("scatter_plot"),
                    full_screen=True
                )
            ),
            col_widths=[6, 8]
        )
    ),
    
    # 4) Cumulative Return Comparison
    ui.card(
        ui.card_header("Cumulative Return Comparison"),
        ui.output_plot("cumulative_return_plot"),
        full_screen=True
    )
)

# 8. Shiny App: Server
def server(input, output, session):
    # Plot covariance heatmap
    @output
    @render.plot
    def heatmap_plot():
        plt.figure(figsize=(4, 3))
        sns.heatmap(cov_mat_annual, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Annualized Covariance Matrix Heatmap")
        return plt.gcf()

    # Strategy selection
    @reactive.Calc
    @reactive.event(input.apply_button)
    def chosen_strategy():
        return input.portfolio_strategy()

    # Strategy weights table
    @output
    @render.table
    def strategy_table():
        selected = chosen_strategy()
        if not selected:
            return pd.DataFrame()
        
        # Prepare weight DataFrames for each strategy
        df_gmv = GMV_weights_df.copy()
        df_gmv.columns = ["Code", "GMV Weight"]
        
        df_maxvar = MaxVar_weights_df.copy()
        df_maxvar.columns = ["Code", "Max Var Weight"]
        
        df_msr = MSR_weights_df.copy()
        df_msr.columns = ["Code", "MSR Weight"]
        
        # Merge based on selection
        if len(selected) == 1:
            if "GMV" in selected:
                return df_gmv
            elif "Max Variance" in selected:
                return df_maxvar
            elif "MSR" in selected:
                return df_msr
        else:
            df_merged = pd.DataFrame({"Code": code_list})
            if "GMV" in selected:
                df_merged = df_merged.merge(df_gmv, on="Code", how="left")
            if "Max Variance" in selected:
                df_merged = df_merged.merge(df_maxvar, on="Code", how="left")
            if "MSR" in selected:
                df_merged = df_merged.merge(df_msr, on="Code", how="left")
            return df_merged

    # Plot scatter (random portfolios + efficient frontier + CAL)
    @output
    @render.plot
    def scatter_plot():
        selected = chosen_strategy()
        plt.figure(figsize=(6, 4))
        # 1) Random portfolios scatter
        plt.scatter(RandomPortfolios["Volatility"], RandomPortfolios["Returns"],
                    alpha=0.3, label="Random Portfolios")

                # 2) If GMV is selected, mark the minimum volatility portfolio from random portfolios (for reference)
        if "GMV" in selected and not positive_returns.empty:
            x_gmv_rand = RandomPortfolios.loc[min_positive_index, "Volatility"]
            y_gmv_rand = RandomPortfolios.loc[min_positive_index, "Returns"]
            plt.scatter(x_gmv_rand, y_gmv_rand, color="red", s=60, label="GMV (Rand)")

        # 3) If Max Variance is selected, mark the maximum volatility portfolio from random portfolios
        if "Max Variance" in selected and not positive_returns.empty:
            x_maxvar = RandomPortfolios.loc[max_positive_index, "Volatility"]
            y_maxvar = RandomPortfolios.loc[max_positive_index, "Returns"]
            plt.scatter(x_maxvar, y_maxvar, color="green", s=60, label="Max Variance (Rand)")

        # 4) Compute and plot the "true" efficient frontier via quadratic programming
        frontier_df = compute_efficient_frontier(mu_np, Sigma_np, short=False, points=50)
        plt.plot(frontier_df["Volatility"], frontier_df["Returns"],
                 color="blue", lw=2, label="Efficient Frontier")

        # 5) Plot MSR (tangency portfolio) + CAL
        msr_ret = msr_weights @ mu_np
        msr_vol = np.sqrt(msr_weights @ Sigma_np @ msr_weights)
        sharpe_ratio = (msr_ret - risk_free) / msr_vol
        
        # Plot CAL (from 0 to 1.5 times MSR volatility)
        cal_x = np.linspace(0, msr_vol * 1.5, 50)
        cal_y = risk_free + sharpe_ratio * cal_x
        plt.plot(cal_x, cal_y, "--", color="purple", label=f"CAL (Sharpe={sharpe_ratio:.2f})")

        # If MSR is selected, mark the MSR point
        if "MSR" in selected:
            plt.scatter(msr_vol, msr_ret, color="orange", s=80, label="MSR (Tangency)")

        plt.xlabel("Volatility")
        plt.ylabel("Annual Return")
        plt.title("Portfolios (GMV / Max Variance / MSR) with Efficient Frontier and CAL")
        plt.legend()
        return plt.gcf()

    # Plot custom target portfolio scatter
    @reactive.Calc
    def custom_portfolio():
        target_r = input.target_return()
        target_v = input.target_volatility()
        dist = ((RandomPortfolios["Returns"] - target_r)**2 + 
                (RandomPortfolios["Volatility"] - target_v)**2).apply(np.sqrt)
        idx_min = dist.idxmin()
        return RandomPortfolios.iloc[idx_min]

    @output
    @render.plot
    def target_scatter_plot():
        row = custom_portfolio()
        x_target = row["Volatility"]
        y_target = row["Returns"]
        plt.figure(figsize=(5, 3))
        plt.scatter(RandomPortfolios["Volatility"], RandomPortfolios["Returns"],
                    alpha=0.3, label="Random Portfolios")
        plt.scatter(x_target, y_target, color="red", s=60, label="Target Portfolio")
        plt.xlabel("Volatility")
        plt.ylabel("Returns")
        plt.title("Target Portfolio")
        plt.legend()
        return plt.gcf()

    @output
    @render.table
    def target_weight_table():
        row = custom_portfolio()
        w_ = row.iloc[0:n].values
        df_ = pd.DataFrame({"Code": code_list, "Weight": w_})
        df_["Returns"] = row["Returns"]
        df_["Volatility"] = row["Volatility"]
        return df_

    # Download strategy weights
    def get_strategy_df():
        selected = chosen_strategy()
        if not selected:
            return pd.DataFrame()
        
        df_gmv = GMV_weights_df.copy()
        df_gmv.columns = ["Code", "GMV Weight"]
        
        df_maxvar = MaxVar_weights_df.copy()
        df_maxvar.columns = ["Code", "Max Var Weight"]
        
        df_msr = MSR_weights_df.copy()
        df_msr.columns = ["Code", "MSR Weight"]

        df_final = pd.DataFrame({"Code": code_list})
        if "GMV" in selected:
            df_final = df_final.merge(df_gmv, on="Code", how="left")
        if "Max Variance" in selected:
            df_final = df_final.merge(df_maxvar, on="Code", how="left")
        if "MSR" in selected:
            df_final = df_final.merge(df_msr, on="Code", how="left")
        return df_final

    @render.download(filename="Strategy_Weights.xlsx")
    def download_weights():
        df_ = get_strategy_df()
        file_path = os.path.join(os.path.dirname(__file__), "Strategy_Weights.xlsx")
        df_.to_excel(file_path, index=False)
        return file_path

    # Display "Strategy applied!" message
    @reactive.effect
    @reactive.event(input.apply_button)
    def show_message():
        m = ui.modal("Strategy applied!", easy_close=True, footer=None)
        ui.modal_show(m)

    # Cumulative return plot
    @output
    @render.plot
    def cumulative_return_plot():
        plt.figure(figsize=(10, 6))
        # Here display four lines: market cap weighted, equal weighted, GMV, MSR
        plt.plot(StockReturns.index, cum_ret_marketcap[1:], color="hotpink", label="MarketcapWeighted")
        plt.plot(StockReturns.index, cum_ret_equal[1:], color="cornflowerblue", label="EqualWeighted")
        plt.plot(StockReturns.index, cum_ret_gmv[1:], color="deepskyblue", label="GMV")
        plt.plot(StockReturns.index, cum_ret_msr[1:], color="red", label="MSR")
        
        # Add horizontal line for initial capital level
        plt.axhline(y=initial_amount, color="black", linestyle="--", label="Initial Amount")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend(loc="upper left", fontsize=12)
        plt.title("Cumulative Return Comparison")
        return plt.gcf()

app = App(app_ui, server)


'''
# Reference

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


'''