# Pair-trading_IEFxFut
**A Pair Trading Strategy Using IEF and ZN with ARIMA Residuals**


**Abstract:**

This paper aims to study a pair trading strategy using the iShares Barclays 7-10 Year Treasury Bond Fund (NYSE: IEF) and 10-year Treasury Note futures (ZN). The approach combines an Ordinary Least Squares (OLS) regression model and an ARIMA model for the residuals, leveraging both models to generate trading signals. The strategy's effectiveness is evaluated based on various performance metrics, including the Sharpe ratio and Maximum Drawdown.

**Introduction:**

Pair trading is a market-neutral strategy that exploits the historical correlation between two securities. This paper outlines a method using IEF and ZN, both of which have a strong inherent relationship as they are both connected to the US Treasury market.

**Methodology:**

1. **Data Acquisition:**
   Data for IEF and ZN was obtained from Yahoo Finance, ranging from January 1, 2000, to October 6, 2023.

2. **OLS Regression:**
   The percentage change in IEF's price was regressed against the percentage change in ZN's price using an OLS regression model to find a linear relationship between the two.

3. **ARIMA Modeling:**
   The residuals from the OLS model were modeled using an ARIMA(1,1,2) to capture any autoregressive and moving average patterns present.

4. **Trading Signals:**
   Signals were generated based on the residuals. When the residuals were above a predefined threshold, a long signal was triggered for IEF and a short for ZN, and vice versa.

5. **Performance Metrics:**
   Several metrics like the Sharpe ratio, maximum drawdown, win/loss ratio, average wins, average losses, and total portfolio return were computed to evaluate the strategy's effectiveness.

**Results:**

1. **OLS Regression Outcome:**
   The regression outcome helps understand the relationship between IEF's and ZN's percentage changes. The residuals provide insight into how much IEF's percentage change deviates from its predicted value based on ZN's percentage change.

2. **Residuals Behavior:**
   The residuals' behavior was observed in-sample (training data) and out-of-sample (test data). The ARIMA(1,1,2) fit on the residuals indicated the presence of autoregressive and moving average components.

3. **Performance:**
   Visualizations provided insights into the strategy's daily performance, allowing for a qualitative evaluation of its efficacy. Performance metrics, such as the Sharpe ratio, win/loss ratio, and maximum drawdown, offered a quantitative assessment.

4. **Forecasting:**
   Utilizing the ARIMA model, residuals for the next three periods were forecasted. This provides a look-ahead for potential trading signals.

**Conclusion:**

The combination of OLS and ARIMA models offers a robust mechanism for generating pair trading signals using IEF and ZN. The strategy's performance metrics highlight its potential benefits and pitfalls. While the backtested results appear promising, real-world trading might involve additional factors like transaction costs and slippage. This approach provides a foundation upon which further refinements and improvements can be made.
