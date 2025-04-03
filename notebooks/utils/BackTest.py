import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd
from tqdm import tqdm
import numpy as np

# -----------------------------------------------------------------------
# Position, Trade, Order Classes
# -----------------------------------------------------------------------
class Position:
    """
    Represents a net trading position for a specific ticker.
    'size' can be positive (long) or negative (short).
    """
    def __init__(self, ticker, size, entry_price, entry_trade):
        self.ticker = ticker
        self.size = size      # net size: +1, +2, ... up to +5; or -1, -2, ... down to -5
        self.entry_price = entry_price
        self.entry_trade = entry_trade

    def __repr__(self):
        return f"<Position: {self.ticker} size: {self.size} entry: {self.entry_price}>"

class Trade:
    """
    Trade objects represent completed transactions.
    """
    def __init__(self, ticker, side, size, price, order_type, idx, role="entry", pnl=0):
        self.ticker = ticker
        self.side = side   # 'buy' or 'sell'
        self.size = size
        self.price = price
        self.order_type = order_type
        self.idx = idx
        self.role = role   # 'entry', 'exit', 'cover', 'reverse'
        self.pnl = pnl     # Profit/Loss for this trade

    def __repr__(self):
        return f"<Trade: {self.idx} {self.ticker} {self.side} {self.size}@{self.price} Role:{self.role} PnL:{self.pnl}>"

class Order:
    """
    Order objects represent intended transactions.
    - side: 'buy' or 'sell'
    - role: 'entry' (in this version, no more 'take_profit'/'stop_loss')
    - order_type: 'market' or 'limit'
    - persistent: always False in this version (no more persistent TP/SL)
    """
    def __init__(self, ticker, side, size, idx, limit_price=None, order_type='market',
                 role='entry', persistent=False):
        self.ticker = ticker
        self.side = side
        self.size = size
        self.order_type = order_type
        self.idx = idx
        self.limit_price = limit_price
        self.role = role  # 'entry'
        self.persistent = persistent  # no-op now
        self.parent_position = None   # no-op now

    def __repr__(self):
        return (f"<Order: {self.idx} {self.ticker} {self.side} {self.size} "
                f"{self.order_type} Role:{self.role} Persistent:{self.persistent}>")

# -----------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------
class Engine:
    def __init__(self, initial_cash=100_000, risk_free_rate=0, transaction_cost=0.001, asset_type='equities'):
        self.strategy = None
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.data = None
        self.current_idx = None
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.asset_type = asset_type

        # We'll track each day's (or bar's) portfolio value
        self.cash_series = {}
        self.stock_series = {}

        self.portfolio = None      # will store final DF with 'stock', 'cash', 'total_aum'
        self.portfolio_bh = None   # buy-and-hold comparison
        self.trading_days = 252 if asset_type == 'equities' else 365

    def add_data(self, data: pd.DataFrame):
        self.data = data

    def add_strategy(self, strategy):
        self.strategy = strategy

    def run(self):
        self.strategy.data = self.data
        self.strategy.cash = self.cash

        for idx in tqdm(self.data.index):
            self.current_idx = idx
            self.strategy.current_idx = self.current_idx

            # Fill any outstanding orders from previous bar
            self._fill_orders()

            # Execute strategy logic on this bar
            self.strategy.on_bar()

            # In this version, we no longer do dynamic TP/SL updates, but you can keep the call if you want:
            self.update_exit_orders()

            # Update the portfolio values
            total_position_value = 0
            for pos in self.strategy.positions:
                # Use the bar's closing price to compute position value
                price = self.data.loc[self.current_idx]['Close']
                total_position_value += pos.size * price

            # Save portfolio states
            self.cash_series[idx] = self.cash
            self.stock_series[idx] = total_position_value

        # Final performance stats
        return self._get_stats(self.asset_type)

    def _fill_orders(self):
        """
        Process all outstanding orders placed by the Strategy in the previous bar:
        - For 'buy' orders, if there's enough cash, we update or create the position.
            Net size is capped at 2x leverage of the initial cash. If the cap is hit, the order is ignored.
        - For 'sell' orders, we update or create the position in the negative direction.
            Net size is capped at -2x leverage of the initial cash. If the cap is hit, the order is ignored.
        """
        remaining_orders = []
        open_price = self.data.loc[self.current_idx]['Open']

        # Total portfolio value with 2x leverage
        max_position_value = 2 * self.initial_cash  # 2x leverage on initial cash

        # Define the cap as the maximum position size based on this leverage
        max_position_size = max_position_value / open_price  # Max BTC you can hold

        for order in self.strategy.orders:
            fill_price = open_price
            can_fill = True  

            if can_fill:
                pos = self.strategy.get_position(order.ticker)
                transaction_fee = fill_price * order.size * self.transaction_cost

                if order.side == 'buy':
                    current_size = pos.size if pos else 0
                    new_size = current_size + order.size

                    if new_size > max_position_size:  # Ignore order if it exceeds leverage cap
                        continue

                    if pos and pos.size < 0:  # Closing a short position
                        size_to_cover = min(order.size, abs(pos.size))
                        pnl = (pos.entry_price - fill_price) * size_to_cover  # Short: Profit when buy price is lower
                        self.cash -= (fill_price * size_to_cover + transaction_fee)

                        self.strategy.trades.append(
                            Trade(order.ticker, 'buy', size_to_cover, fill_price, order.order_type,
                                self.current_idx, "cover", pnl)
                        )

                        if size_to_cover == abs(pos.size):
                            self.strategy.positions.remove(pos)
                        else:
                            pos.size += size_to_cover  # Reduce short

                    elif pos is None or pos.size >= 0:  # New long or adding to long
                        cost = fill_price * order.size + transaction_fee
                        if self.cash >= cost:
                            self.cash -= cost
                            if pos:
                                pos.size += order.size
                            else:
                                pos = Position(order.ticker, order.size, fill_price, None)
                                self.strategy.positions.append(pos)

                            self.strategy.trades.append(
                                Trade(order.ticker, 'buy', order.size, fill_price, order.order_type,
                                    self.current_idx, "entry")
                            )

                elif order.side == 'sell':
                    current_size = pos.size if pos else 0
                    new_size = current_size - order.size

                    if new_size < -max_position_size:  # Ignore order if it exceeds leverage cap on short positions
                        continue

                    if pos and pos.size > 0:  # Closing a long position
                        size_to_sell = min(order.size, pos.size)
                        pnl = (fill_price - pos.entry_price) * size_to_sell  # Long: Profit when sell price is higher
                        self.cash += (fill_price * size_to_sell - transaction_fee)

                        self.strategy.trades.append(
                            Trade(order.ticker, 'sell', size_to_sell, fill_price, order.order_type,
                                self.current_idx, "exit", pnl)
                        )

                        if size_to_sell == pos.size:
                            self.strategy.positions.remove(pos)
                        else:
                            pos.size -= size_to_sell  # Reduce long

                    elif pos is None or pos.size <= 0:  # New short or adding to short
                        proceeds = fill_price * order.size - transaction_fee
                        self.cash += proceeds
                        if pos:
                            pos.size -= order.size
                        else:
                            pos = Position(order.ticker, -order.size, fill_price, None)
                            self.strategy.positions.append(pos)

                        self.strategy.trades.append(
                            Trade(order.ticker, 'sell', order.size, fill_price, order.order_type,
                                self.current_idx, "entry")
                        )

        self.strategy.orders = remaining_orders


    def update_exit_orders(self):
        """
        In the old code, you updated TP/SL dynamically here.
        Now it's a no-op, but you can keep it if you want to do
        other tasks each bar.
        """
        pass

    def _get_stats(self, asset_type='equities'):
        """
        Compute final metrics and portfolio stats.
        """
        metrics = {}

        print(f"Initial Portfolio Value: {self.initial_cash}")

        # Final portfolio value includes net positions at final close:
        total_position_value = 0
        final_price = self.data.loc[self.current_idx, 'Close']
        for pos in self.strategy.positions:
            total_position_value += pos.size * final_price

        final_value = self.cash + total_position_value

        print(f"Final Portfolio Value: {final_value}")

        # Transaction costs are already subtracted from self.cash whenever we traded,
        # so final_value is already net of transaction costs in this simplified approach.
        total_return = 100.0 * (final_value / self.initial_cash - 1.0)
        metrics['Total Return (%)'] = total_return

        # Build daily portfolio time series
        portfolio = pd.DataFrame({'stock': self.stock_series, 'cash': self.cash_series})
        portfolio['total_aum'] = portfolio['stock'] + portfolio['cash']
        self.portfolio = portfolio

        # Buy-and-hold: invests all initial cash at first open price
        initial_price = self.data.iloc[0]['Open']
        transaction_fee = self.initial_cash * self.transaction_cost
        initial_investment = self.initial_cash - transaction_fee
        portfolio_bh = (initial_investment / initial_price) * self.data['Close']

        self.portfolio_bh = portfolio_bh

        buy_and_hold_final_value = portfolio_bh.iloc[-1]

        # Subtract a transaction cost when "selling" at the end
        final_transaction_cost = buy_and_hold_final_value * self.transaction_cost
        buy_and_hold_final_value -= final_transaction_cost

        print(f"Buy & Hold Final Value: {buy_and_hold_final_value}")

        buy_and_hold_return = 100.0 * (buy_and_hold_final_value / self.initial_cash - 1.0)
        metrics['Buy-and-Hold Total Return (%)'] = buy_and_hold_return

        # Average exposure to asset
        portfolio['pct_exposure'] = 100.0 * portfolio['stock'] / portfolio['total_aum']
        metrics['Average Exposure to Asset (%)'] = portfolio['pct_exposure'].mean()

        # Strategy CAGR
        p = portfolio['total_aum']
        days_diff = (p.index[-1] - p.index[0]).days if isinstance(p.index[-1], pd.Timestamp) else (p.index[-1] - p.index[0])
        years = days_diff / self.trading_days if self.trading_days else 1
        if years <= 0:
            years = 1
        cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
        metrics['Strategy CAGR (%)'] = 100.0 * cagr

        # B&H CAGR
        p_bh = portfolio_bh
        days_diff_bh = (p_bh.index[-1] - p_bh.index[0]).days if isinstance(p_bh.index[-1], pd.Timestamp) else (p_bh.index[-1] - p_bh.index[0])
        years_bh = days_diff_bh / self.trading_days if self.trading_days else 1
        if years_bh <= 0:
            years_bh = 1
        cagr_bh = (p_bh.iloc[-1] / p_bh.iloc[0]) ** (1 / years_bh) - 1
        metrics['Buy & Hold CAGR (%)'] = 100.0 * cagr_bh

        # Annualized volatility
        strategy_vol = p.pct_change().std() * np.sqrt(self.trading_days) * 100
        bh_vol = p_bh.pct_change().std() * np.sqrt(self.trading_days) * 100
        metrics['Strategy Volatility (%)'] = strategy_vol
        metrics['Buy & Hold Volatility (%)'] = bh_vol

        # Sharpe Ratio
        rf = self.risk_free_rate
        metrics['Strategy Sharpe Ratio'] = (metrics['Strategy CAGR (%)'] - rf) / (strategy_vol if strategy_vol else 1e-9)
        metrics['Buy & Hold Sharpe Ratio'] = (metrics['Buy & Hold CAGR (%)'] - rf) / (bh_vol if bh_vol else 1e-9)

        # Max Drawdowns
        metrics['Strategy Max Drawdown (%)'] = self.get_max_drawdown(p)
        metrics['Buy & Hold Max Drawdown (%)'] = self.get_max_drawdown(p_bh)

        # Number of trades
        metrics['Number of Trades'] = len(self.strategy.trades)
        metrics['Number of Buys'] = len([t for t in self.strategy.trades if t.side == 'buy'])
        metrics['Number of Sells'] = len([t for t in self.strategy.trades if t.side == 'sell'])

        # Compute Win Rate: consider trades that have closed the position ("exit" and "cover")
        closed_trades = [trade for trade in self.strategy.trades if trade.role in ['exit', 'cover']]
        if closed_trades:
            winning_trades = sum(1 for trade in closed_trades if trade.pnl > 0)
            win_rate = (winning_trades / len(closed_trades)) * 100.0
        else:
            win_rate = 0
        metrics['Win Rate (%)'] = round(win_rate, 2)

        if closed_trades:
            avg_profit = sum(trade.pnl for trade in closed_trades) / len(closed_trades)
        else:
            avg_profit = 0

        metrics['Avg Profit per Trade'] = round(avg_profit, 2)

        self.print_metrics(metrics)
        return metrics


    @staticmethod
    def get_max_drawdown(series):
        roll_max = series.cummax()
        daily_drawdown = series / roll_max - 1.0
        max_drawdown = daily_drawdown.cummin().min() * 100
        return max_drawdown

    def print_metrics(self, metrics):
        print(f"{'Metric':40} | {'Value':>10}")
        print("-" * 55)
        for k, v in metrics.items():
            print(f"{k:40} | {v:10.2f}")

    def plot(self, show_signals=True):
        fig = go.Figure()

        # Strategy AUM
        fig.add_trace(go.Scatter(
            x=self.portfolio.index,
            y=self.portfolio['total_aum'],
            mode='lines',
            name='Strategy'
        ))

        # Buy & Hold
        fig.add_trace(go.Scatter(
            x=self.portfolio_bh.index,
            y=self.portfolio_bh,
            mode='lines',
            name='Buy & Hold'
        ))

        if show_signals and hasattr(self.strategy, 'trades'):
            trades = self.strategy.trades
            buy_trades = [(t.idx, t.price) for t in trades if t.side == 'buy']
            sell_trades = [(t.idx, t.price) for t in trades if t.side == 'sell']

            if len(buy_trades) > 0:
                buy_dates, buy_prices = zip(*buy_trades)
                fig.add_trace(go.Scatter(
                    x=buy_dates, y=buy_prices,
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=10)
                ))

            if len(sell_trades) > 0:
                sell_dates, sell_prices = zip(*sell_trades)
                fig.add_trace(go.Scatter(
                    x=sell_dates, y=sell_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=10)
                ))

        fig.update_layout(
            title='Strategy vs. Buy & Hold',
            xaxis_title='Date',
            yaxis_title='Portfolio Value',
            hovermode='x unified',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=24, label='1d', step='hour', stepmode='backward'),
                        dict(count=72, label='3d', step='hour', stepmode='backward'),
                        dict(step='all', label='All')
                    ])
                ),
                rangeslider=dict(visible=True),
                type='date'
            )
        )
        fig.show()

# -----------------------------------------------------------------------
# Strategy (Base Class)
# -----------------------------------------------------------------------
class Strategy:
    """
    Base strategy class.
    We now rely solely on 'buy'/'sell' signals for opening/increasing/decreasing positions.
    """
    def __init__(self):
        self.current_idx = None
        self.data = None
        self.cash = None
        self.orders = []       # fresh each bar; filled in the next bar
        self.trades = []       # completed trades
        self.positions = []    # open net positions (one per ticker)

    def close(self):
        # convenience to return the closing price
        return self.data.loc[self.current_idx, 'Close']

    def get_position(self, ticker):
        # Return the open position for the ticker (if any). Single net position approach.
        for pos in self.positions:
            if pos.ticker == ticker:
                return pos
        return None
    
    def get_position_size(self, ticker):
        """
        Computes target position size as 10% of total portfolio value.
        Total portfolio value = current cash + market value of current position for the ticker.
        """
        current_price = self.close()  # assuming this is the price for ticker; adjust if necessary
        # Calculate the market value of your current position in the given ticker (if any)
        pos = self.get_position(ticker)
        current_position_value = pos.size * current_price if pos is not None else 0
        return current_position_value

    def buy(self, ticker, size=1, limit_price=None):
        """
        Issue a buy order. This will move the net position up by `size`.
        If position is negative (short), this will partially close the short, or
        even flip to long if size is big enough.
        """
        order_type = 'market' if limit_price is None else 'limit'
        self.orders.append(Order(
            ticker=ticker,
            side='buy',
            size=size,
            idx=self.current_idx,
            limit_price=limit_price,
            order_type=order_type,
            role='entry',
            persistent=False
        ))

    def sell(self, ticker, size=1, limit_price=None):
        """
        Issue a sell order. This will move the net position down by `size`.
        If position is positive (long), this will reduce or close the long,
        or even flip to short if size is big enough.
        """
        order_type = 'market' if limit_price is None else 'limit'
        self.orders.append(Order(
            ticker=ticker,
            side='sell',
            size=size,
            idx=self.current_idx,
            limit_price=limit_price,
            order_type=order_type,
            role='entry',
            persistent=False
        ))

    def on_bar(self):
        """
        Override this method with your strategy logic.
        For example, you check signals and decide to buy or sell.
        """
        raise NotImplementedError("You must implement on_bar() in your concrete strategy.")

    def update_exit_orders(self, current_idx):
        """
        No-op in this simplified version, since we are not using TP/SL anymore.
        """
        pass

