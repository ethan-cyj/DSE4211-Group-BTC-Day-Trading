import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 12]

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd 
from tqdm import tqdm 
import numpy as np

class Position:
    """Represents an open trading position with associated take profit and stop loss orders."""
    def __init__(self, ticker, size, entry_price, entry_trade):
        self.ticker = ticker
        self.size = size
        self.entry_price = entry_price
        self.entry_trade = entry_trade
        self.tp_order = None
        self.sl_order = None

    def __repr__(self):
        return f"<Position: {self.ticker} size: {self.size} entry: {self.entry_price}>"

class Engine():
    def __init__(self, initial_cash=100_000, risk_free_rate=0, asset_type='equities'):
        self.strategy = None 
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.data = None 
        self.current_idx = None 
        self.risk_free_rate = risk_free_rate
        self.asset_type = asset_type
        self.cash_series = {}
        self.stock_series = {}

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
            # Fill orders from previous periods (persistent TP/SL orders are retained)
            self._fill_orders()

            # Run the strategy on the current bar 
            self.strategy.on_bar()

            # Update exit orders (TP/SL) based on latest volatility 
            self.update_exit_orders()

            # Update portfolio values: sum over all open positions
            total_position_value = sum([pos.size * self.data.loc[self.current_idx]['Close'] for pos in self.strategy.positions])
            self.cash_series[idx] = self.cash
            self.stock_series[idx] = total_position_value

            # --- Print portfolio info ---
            try:
                price = self.data.loc[self.current_idx]['Close']
                total_value = self.cash + total_position_value
                #print(f"Index: {idx} | Cash: {self.cash:.2f} | Total position value: {total_position_value:.2f} | Total value: {total_value:.2f}")
            except Exception as e:
                print(f"Error at index {idx}: {e}")

        return self._get_stats(self.asset_type)

    def _fill_orders(self):
        """
        Process orders based on current bar data.
        
        - Entry buy orders are filled if cash is sufficient and fewer than 5 positions are open.
        - When an entry order fills, a new position is created and two persistent TP/SL orders (default 5% levels) 
          are automatically generated and added.
        - For TP/SL orders, if the market price meets the condition, the order fills and the corresponding 
          position is closed.
        - Orders that are persistent (TP/SL) remain in the order list until filled.
        """
        remaining_orders = []
        for order in self.strategy.orders:
            fill_price = self.data.loc[self.current_idx]['Open']
            can_fill = False

            # --- Entry Buy Orders ---
            if order.side == 'buy' and order.role == 'entry':
                if self.cash >= self.data.loc[self.current_idx]['Open'] * order.size and len(self.strategy.positions) < 20:
                    if order.order_type == "limit":
                        if order.limit_price >= self.data.loc[self.current_idx]['Low']:
                            fill_price = order.limit_price
                            can_fill = True
                            print(f"{self.current_idx} Buy Entry Filled. Limit: {order.limit_price} / Low: {self.data.loc[self.current_idx]['Low']}")
                        else:
                            print(f"{self.current_idx} Buy Entry Not Filled. Limit: {order.limit_price} / Low: {self.data.loc[self.current_idx]['Low']}")
                    else:
                        can_fill = True
                else:
                    if len(self.strategy.positions) >= 10:
                        print(f"{self.current_idx} Buy Entry Not Filled. Maximum positions reached (10).")
                    else:
                        print(f"{self.current_idx} Buy Entry Not Filled. Insufficient cash.")
            
            # --- Manual Exit (Entry Sell Orders) ---
            elif order.side == 'sell' and order.role == 'entry':
                pos = self.strategy.get_position(order.ticker)
                if pos is not None:
                    if order.order_type == "limit":
                        if self.data.loc[self.current_idx]['High'] >= order.limit_price:
                            fill_price = order.limit_price
                            can_fill = True
                            print(f"{self.current_idx} Manual Exit Filled. Limit: {order.limit_price} / High: {self.data.loc[self.current_idx]['High']}")
                        else:
                            print(f"{self.current_idx} Manual Exit Not Filled. Limit: {order.limit_price} / High: {self.data.loc[self.current_idx]['High']}")
                    else:
                        can_fill = True
                else:
                    print(f"{self.current_idx} Manual Exit Order ignored. No open position for ticker {order.ticker}.")

            # --- Take Profit and Stop Loss Orders (Sell Orders to close position) ---
            elif order.side == 'sell' and order.role in ['take_profit', 'stop_loss']:
                pos = order.parent_position
                # Only process if the position is still open
                if pos in self.strategy.positions:
                    if order.order_type == "limit":
                        if order.role == 'take_profit':
                            if self.data.loc[self.current_idx]['High'] >= order.limit_price:
                                fill_price = order.limit_price
                                can_fill = True
                                print(f"{self.current_idx} Take Profit Filled. Limit: {order.limit_price} / High: {self.data.loc[self.current_idx]['High']}")
                            # else:
                            #     print(f"{self.current_idx} Take Profit Not Filled. Limit: {order.limit_price} / High: {self.data.loc[self.current_idx]['High']}")
                        elif order.role == 'stop_loss':
                            if self.data.loc[self.current_idx]['Low'] <= order.limit_price:
                                fill_price = order.limit_price
                                can_fill = True
                                print(f"{self.current_idx} Stop Loss Filled. Limit: {order.limit_price} / Low: {self.data.loc[self.current_idx]['Low']}")
                            # else:
                            #     print(f"{self.current_idx} Stop Loss Not Filled. Limit: {order.limit_price} / Low: {self.data.loc[self.current_idx]['Low']}")
                    else:
                        can_fill = True
                else:
                    # Position already closed, skip order
                    continue

            if can_fill:
                # Create trade object (recording the fill)
                t = Trade(
                    ticker=order.ticker,
                    side=order.side,
                    size=order.size,
                    price=fill_price,
                    order_type=order.order_type,
                    idx=self.current_idx,
                    role=order.role
                )
                self.strategy.trades.append(t)

                if order.side == 'buy' and order.role == 'entry':
                    self.cash -= fill_price * order.size
                    # Open a new position
                    pos = Position(ticker=order.ticker, size=order.size, entry_price=fill_price, entry_trade=t)
                    self.strategy.positions.append(pos)
                    print(f"{self.current_idx} New position opened: {pos}")

                    # Calculate dynamic TP/SL levels based on volatility (e.g. ATR)
                    tp_price, sl_price = self.strategy.calculate_tp_sl(fill_price, self.current_idx)
                    tp_order = Order(
                        ticker=order.ticker,
                        side='sell',
                        size=order.size,
                        idx=self.current_idx,
                        limit_price=tp_price,
                        order_type='limit',
                        role='take_profit',
                        persistent=True
                    )
                    sl_order = Order(
                        ticker=order.ticker,
                        side='sell',
                        size=order.size,
                        idx=self.current_idx,
                        limit_price=sl_price,
                        order_type='limit',
                        role='stop_loss',
                        persistent=True
                    )
                    # Link these orders to the newly opened position
                    tp_order.parent_position = pos
                    sl_order.parent_position = pos
                    pos.tp_order = tp_order
                    pos.sl_order = sl_order
                    remaining_orders.extend([tp_order, sl_order])
                elif order.side == 'sell':
                    # For exit orders (manual exit or TP/SL), close the corresponding position
                    pos = None
                    if order.role in ['take_profit', 'stop_loss']:
                        pos = order.parent_position
                    elif order.role == 'entry':
                        pos = self.strategy.get_position(order.ticker)
                    if pos is not None and pos in self.strategy.positions:
                        self.strategy.positions.remove(pos)
                        self.cash += fill_price * abs(pos.size)
                        print(f"{self.current_idx} Position closed for ticker {pos.ticker} at {fill_price}")
                        # Do not add associated persistent orders (TP/SL) back
                # Order is filled so we do not add it again (persistent orders that are filled get removed)
            else:
                # For persistent orders (TP/SL), keep them for future ticks.
                if order.persistent:
                    remaining_orders.append(order)
                # Non-persistent orders that are not filled are dropped.
        self.strategy.orders = remaining_orders
    
    def update_exit_orders(self):
        """Endpoint to update TP/SL orders dynamically based on latest volatility."""
        self.strategy.update_exit_orders(self.current_idx)

    def _get_stats(self, asset_type='equities'):
        metrics = {}

        # Final return calculation: cash plus value of open positions
        total_position_value = sum([pos.size * self.data.loc[self.current_idx]['Close'] for pos in self.strategy.positions])
        final_value = total_position_value + self.cash
        total_return = 100 * ((final_value / self.initial_cash) - 1)
        metrics['Total Return (%)'] = total_return

        # Strategy portfolio over time
        portfolio = pd.DataFrame({'stock': self.stock_series, 'cash': self.cash_series})
        portfolio['total_aum'] = portfolio['stock'] + portfolio['cash']
        self.portfolio = portfolio

        # Buy-and-hold strategy (investing all initial cash at first open price)
        initial_price = self.data.loc[self.data.index[0], 'Open']
        portfolio_bh = (self.initial_cash / initial_price) * self.data['Close']
        self.portfolio_bh = portfolio_bh

        # Average exposure to asset
        metrics['Average Exposure to Asset (%)'] = ((portfolio['stock'] / portfolio['total_aum']) * 100).mean()

        # Strategy CAGR
        p = portfolio['total_aum']
        days_diff = (p.index[-1] - p.index[0]).days if isinstance(p.index[-1], pd.Timestamp) else (p.index[-1] - p.index[0])
        metrics['Strategy CAGR (%)'] = ((p.iloc[-1] / p.iloc[0]) ** (1 / (days_diff / 365)) - 1) * 100

        # Buy-and-hold CAGR
        p_bh = portfolio_bh
        days_diff_bh = (p_bh.index[-1] - p_bh.index[0]).days if isinstance(p_bh.index[-1], pd.Timestamp) else (p_bh.index[-1] - p_bh.index[0])
        metrics['Buy & Hold CAGR (%)'] = ((p_bh.iloc[-1] / p_bh.iloc[0]) ** (1 / (days_diff_bh / 365)) - 1) * 100

        # Determine trading days based on asset type
        self.trading_days = 252 if asset_type == 'equities' else 365

        # Annualized volatility
        metrics['Strategy Volatility (%)'] = p.pct_change().std() * np.sqrt(self.trading_days) * 100
        metrics['Buy & Hold Volatility (%)'] = portfolio_bh.pct_change().std() * np.sqrt(self.trading_days) * 100

        # Sharpe Ratios
        rf = self.risk_free_rate
        metrics['Strategy Sharpe Ratio'] = (metrics['Strategy CAGR (%)'] - rf) / metrics['Strategy Volatility (%)']
        metrics['Buy & Hold Sharpe Ratio'] = (metrics['Buy & Hold CAGR (%)'] - rf) / metrics['Buy & Hold Volatility (%)']

        # Maximum Drawdowns
        metrics['Strategy Max Drawdown (%)'] = self.get_max_drawdown(p)
        metrics['Buy & Hold Max Drawdown (%)'] = self.get_max_drawdown(portfolio_bh)

        # Number of trades
        metrics['Number of Trades'] = len(self.strategy.trades)
        # Number of Buys
        metrics['Number of Buys'] = len([t for t in self.strategy.trades if t.side == 'buy'])
        # Number of Sells
        metrics['Number of Sells'] = len([t for t in self.strategy.trades if t.side == 'sell'])
        # Number of times took profit
        metrics['Number of Take Profits'] = len([t for t in self.strategy.trades if t.role == 'take_profit'])
        # Number of times stopped loss
        metrics['Number of Stop Losses'] = len([t for t in self.strategy.trades if t.role == 'stop_loss'])

        # Print formatted results
        self.print_metrics(metrics)

        return metrics

    def print_metrics(self, metrics):
        print(f"{'Metric':40} | {'Value':>10}")
        print("-" * 55)
        for key, value in metrics.items():
            print(f"{key:40} | {value:10.2f}")

    @staticmethod
    def get_max_drawdown(close):
        roll_max = close.cummax()
        daily_drawdown = close / roll_max - 1.0
        max_daily_drawdown = daily_drawdown.cummin()
        return max_daily_drawdown.min() * 100

    def plot(self, show_signals=True):
        fig = go.Figure()

        # Strategy AUM
        fig.add_trace(go.Scatter(
            x=self.portfolio.index,
            y=self.portfolio['total_aum'],
            mode='lines',
            name='Strategy',
            line=dict(color='blue')
        ))

        # Buy & Hold AUM
        fig.add_trace(go.Scatter(
            x=self.portfolio_bh.index,
            y=self.portfolio_bh,
            mode='lines',
            name='Buy & Hold',
            line=dict(dash='dash', color='gray')
        ))

        # Buy/Sell signals from trades
        if show_signals and hasattr(self.strategy, 'trades'):
            trades = self.strategy.trades
            buy_trades = [(t.idx, t.price) for t in trades if t.side == 'buy']
            sell_trades = [(t.idx, t.price) for t in trades if t.side == 'sell']
            if buy_trades:
                buy_dates, buy_prices = zip(*buy_trades)
                fig.add_trace(go.Scatter(
                    x=buy_dates, y=buy_prices,
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    showlegend=False
                ))
            if sell_trades:
                sell_dates, sell_prices = zip(*sell_trades)
                fig.add_trace(go.Scatter(
                    x=sell_dates, y=sell_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    showlegend=False
                ))

        fig.update_layout(
            title='Interactive Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            legend=dict(x=0.01, y=0.99),
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

class Strategy():
    """This base class will handle the execution logic of our trading strategies."""
    def __init__(self):
        self.current_idx = None
        self.data = None
        self.cash = None
        self.orders = []
        self.trades = []
        self.positions = []  # Track open positions
        self.tp_atr_multiplier = 2.0  # Multiplier for TP/SL levels in terms of ATR
        self.sl_atr_multiplier = 1.0  # Multiplier for TP/SL levels in terms of ATR

    def close(self):
        return self.data.loc[self.current_idx]['Close']

    def position_size(self):
        # Sum sizes of all open positions
        return sum([pos.size for pos in self.positions])

    def get_position(self, ticker):
        # Return the first open position for a given ticker (if any)
        for pos in self.positions:
            if pos.ticker == ticker:
                return pos
        return None

    def buy(self, ticker, size=1):
        # Only add an entry order if maximum positions not reached
        if len(self.positions) < 10:
            self.orders.append(
                Order(
                    ticker=ticker,
                    side='buy',
                    size=size,
                    idx=self.current_idx,
                    role='entry',
                    persistent=False
                )
            )
        else:
            print(f"{self.current_idx} Cannot buy {ticker}: maximum positions reached.")

    def sell(self, ticker, size=1):
        # Manual exit order for an existing position
        pos = self.get_position(ticker)
        if pos:
            self.orders.append(
                Order(
                    ticker=ticker,
                    side='sell',
                    size=size,
                    idx=self.current_idx,
                    role='entry',  # using 'entry' role for manual exits
                    persistent=False
                )
            )
        else:
            print(f"{self.current_idx} No open position for ticker {ticker} to sell.")

    def buy_limit(self, ticker, limit_price, size=1):
        self.orders.append(
            Order(
                ticker=ticker,
                side='buy',
                size=size,
                limit_price=limit_price,
                order_type='limit',
                idx=self.current_idx,
                role='entry',
                persistent=False
            )
        )

    def sell_limit(self, ticker, limit_price, size=1):
        self.orders.append(
            Order(
                ticker=ticker,
                side='sell',
                size=size,
                limit_price=limit_price,
                order_type='limit',
                idx=self.current_idx,
                role='entry',
                persistent=False
            )
        )

    def calculate_tp_sl(self, fill_price, current_idx):
        """
        Dynamically calculate TP/SL prices based on a volatility measure.
        Here we assume that your data contains an 'ATR' column.
        If 'ATR' is not present, a default (5% of fill_price) is used.
        """
        atr = self.data.loc[current_idx].get('ATR', fill_price * 0.05)
        predicted_volatility_category =self.data.loc[current_idx].get('volatility_category', 'normal')
        if predicted_volatility_category == 'low':
            self.tp_atr_multiplier = 2.0
            self.sl_atr_multiplier = 1.0
        elif predicted_volatility_category == 'high':
            self.tp_atr_multiplier = 3.5
            self.sl_atr_multiplier = 2.5
        else:
            self.tp_atr_multiplier = 2.5
            self.sl_atr_multiplier = 1.5
        tp_price = fill_price + self.tp_atr_multiplier * atr
        sl_price = fill_price - self.sl_atr_multiplier * atr
        return tp_price, sl_price

    def update_exit_orders(self, current_idx):
        """Update TP/SL orders for all open positions using the latest volatility."""
        for pos in self.positions:
            tp_price, sl_price = self.calculate_tp_sl(pos.entry_price, current_idx)
            pos.tp_order.limit_price = tp_price
            pos.sl_order.limit_price = sl_price

    def on_bar(self):
        """This should be overridden by your custom strategy logic."""
        raise NotImplementedError("Implement your strategy logic in on_bar()")

class Trade():
    """Trade objects are created when an order is filled."""
    def __init__(self, ticker, side, size, price, order_type, idx, role):
        self.ticker = ticker
        self.side = side
        self.size = size
        self.price = price
        self.order_type = order_type
        self.idx = idx
        self.role = role

    def __repr__(self):
        return f'<Trade: {self.idx} {self.ticker} {self.size}@{self.price} Role:{self.role}>'

class Order():
    """
    Order objects represent intended transactions.
    
    Parameters:
      - role: 'entry', 'take_profit', or 'stop_loss'
      - persistent: If True, the order is not cleared after each tick (used for TP/SL orders)
    """
    def __init__(self, ticker, side, size, idx, limit_price=None, order_type='market', role='entry', persistent=False):
        self.ticker = ticker
        self.side = side
        self.size = size
        self.order_type = order_type
        self.idx = idx
        self.limit_price = limit_price
        self.role = role  # 'entry', 'take_profit', or 'stop_loss'
        self.persistent = persistent  # For TP/SL orders
        self.parent_position = None  # Links TP/SL orders to their position

    def __repr__(self):
        return f'<Order: {self.idx} {self.ticker} {self.side} {self.size} {self.order_type} Role:{self.role} Persistent:{self.persistent}>'
