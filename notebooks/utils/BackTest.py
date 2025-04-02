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
    """Trade objects are created when an order is filled."""
    def __init__(self, ticker, side, size, price, order_type, idx, role):
        self.ticker = ticker
        self.side = side       # 'buy' or 'sell'
        self.size = size
        self.price = price
        self.order_type = order_type
        self.idx = idx
        self.role = role       # 'entry' in this simplified version

    def __repr__(self):
        return f"<Trade: {self.idx} {self.ticker} {self.size}@{self.price} Role:{self.role}>"

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
            Net size is capped at +5.
          - For 'sell' orders, we update or create the position in the negative direction.
            Net size is capped at -5.
        """
        remaining_orders = []
        open_price = self.data.loc[self.current_idx]['Open']
        low_price = self.data.loc[self.current_idx]['Low']
        high_price = self.data.loc[self.current_idx]['High']

        for order in self.strategy.orders:
            # Attempt to fill order
            fill_price = open_price
            can_fill = False

            # Check order type (market or limit)
            if order.order_type == 'market':
                # Always fill at the open price
                fill_price = open_price
                can_fill = True
            elif order.order_type == 'limit':
                # For a 'buy limit', fill if the market's Low <= limit_price
                # For a 'sell limit', fill if the market's High >= limit_price
                if order.side == 'buy':
                    if low_price <= order.limit_price:
                        fill_price = order.limit_price
                        can_fill = True
                else:
                    # side == 'sell'
                    if high_price >= order.limit_price:
                        fill_price = order.limit_price
                        can_fill = True

            # If we can fill, compute fees and adjust positions
            if can_fill:
                transaction_fee = fill_price * order.size * self.transaction_cost

                # If it's a buy, we are pushing net position up
                if order.side == 'buy':
                    pos = self.strategy.get_position(order.ticker)

                    if pos is None:
                        # No existing position => create a new one
                        net_desired_size = min(order.size, 5)  # cannot exceed +5
                        cost_to_buy = fill_price * net_desired_size + transaction_fee
                        if self.cash >= cost_to_buy:
                            self.cash -= cost_to_buy
                            new_pos = Position(order.ticker, net_desired_size, fill_price, None)
                            self.strategy.positions.append(new_pos)
                            self.strategy.trades.append(
                                Trade(order.ticker, 'buy', net_desired_size,
                                      fill_price, order.order_type, self.current_idx, order.role)
                            )
                        else:
                            print(f"{self.current_idx} Not enough cash to open new long position.")
                    else:
                        # We already have a position, possibly negative or positive
                        current_size = pos.size
                        net_desired_size = current_size + order.size
                        if net_desired_size > 5:
                            net_desired_size = 5  # cap at +5

                        size_change = net_desired_size - current_size
                        if size_change > 0:
                            # Increase net size
                            cost_to_buy = fill_price * size_change + transaction_fee
                            if self.cash >= cost_to_buy:
                                self.cash -= cost_to_buy
                                pos.size = net_desired_size
                                self.strategy.trades.append(
                                    Trade(order.ticker, 'buy', size_change,
                                          fill_price, order.order_type, self.current_idx, order.role)
                                )
                            else:
                                print(f"{self.current_idx} Not enough cash to buy additional size.")

                        elif size_change < 0:
                            # This effectively closes or reverses part of the position.
                            # size_change < 0 => we are 'selling' in net terms,
                            # but the order is a 'buy' method. Usually you'd handle partial reversing
                            # by calling Strategy.sell(...) to reduce a long or build a short.
                            pass

                # If it's a sell, we are pushing net position down
                elif order.side == 'sell':
                    pos = self.strategy.get_position(order.ticker)

                    if pos is None:
                        # No existing position => open a new short
                        net_desired_size = -min(order.size, 5)  # go negative, up to -5
                        proceeds = fill_price * abs(net_desired_size) - transaction_fee
                        # For a short, you might track margin, etc. We'll assume no additional constraints:
                        self.cash += proceeds
                        new_pos = Position(order.ticker, net_desired_size, fill_price, None)
                        self.strategy.positions.append(new_pos)
                        self.strategy.trades.append(
                            Trade(order.ticker, 'sell', net_desired_size,
                                  fill_price, order.order_type, self.current_idx, order.role)
                        )
                    else:
                        # Already have a position
                        current_size = pos.size
                        net_desired_size = current_size - order.size  # move in negative direction
                        if net_desired_size < -5:
                            net_desired_size = -5  # cap at -5

                        size_change = net_desired_size - current_size
                        if size_change < 0:
                            # Increase the short or reduce a long
                            proceeds = fill_price * abs(size_change) - transaction_fee
                            self.cash += proceeds
                            pos.size = net_desired_size
                            self.strategy.trades.append(
                                Trade(order.ticker, 'sell', size_change,
                                      fill_price, order.order_type, self.current_idx, order.role)
                            )
                        elif size_change > 0:
                            # This effectively closes or reduces a short
                            # but the signal is 'sell'. Typically you'd handle that with a 'buy' call
                            # if you want to reduce short. We'll keep it no-op in this example.
                            pass

                else:
                    # Should never happen (only 'buy' or 'sell')
                    pass

            # If order not filled, and it's not persistent, we simply drop it
            # (in this simplified version, persistent is always False)
            # so we do nothing.

        # Clear out the old orders
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

        # Final portfolio value includes net positions at final close:
        total_position_value = 0
        final_price = self.data.loc[self.current_idx, 'Close']
        for pos in self.strategy.positions:
            total_position_value += pos.size * final_price

        final_value = self.cash + total_position_value

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
        buy_and_hold_return = 100.0 * (buy_and_hold_final_value / initial_investment - 1.0)
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
