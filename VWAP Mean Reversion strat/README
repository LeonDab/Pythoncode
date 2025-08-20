# VWAP Mean Reversion Strategy on SPY
---


## About
This was my first attempt at creating a mean reversion strategy. I decided to go with a VWAP reversion strategy as I've learned its an important level lots of institutions tend to buy and sell from throughout the day. The strategy uses +/- 2 standard deviations from VWAP as entry points and targets a reversion to VWAP as a take profit and has a fixed 20 dollar change as a stop loss.
---

## Reasoning
- 100k account, 100 shares x $20 = $2000 max loss per trade  --> 2% risk (a bit high but tolerable, risk of ruin is small).
- 2 standard deviations SHOULD contain 95% of the move in theory making it a good entry point.
- Chose SPY as I have experience in index futures and wanted to try something different but not worlds apart.
- 4H dataframe to get rid of noise and try make it more of a 'swing' strategy.
- Only one trade per cross above 2std bands to prevent compounding potentially losing positions.
- Output as CSV to view trades as well as all the statistics and visuals provided by VectorBT.

---

## Interesting Points
- While this strategy SEVERELY underperformed a buy and hold strategy I learned some intersting things along the way which is why I decided to keep it.
- Looking at the win rate, Sharpe ratio, expectancy and profit factor the strategy seems promising, however these statistics may be misleading due to a number of factors which is what I wanted to highlight.
- While parameters such as those listed above being solid can be an indicator of a successful strategy it is not a given and they should always be investigated deeper before deploying a strategy with real capital or sinking lots of time into it.  
- VWAP could be tried for different lengths to try optimise the strategy as it currently resets daily which I realise is not the greatest idea.
