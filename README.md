# DSE4211 Group Project: Bitcoin Day Trading Strategy

## Overview

This repository contains the implementation and analysis of an intraday Bitcoin trading strategy augmented by sentiment analysis. Our approach leverages sentiment indicators extracted from diverse digital sources, combined with technical indicators and advanced predictive modeling (XGBoost), to achieve robust trading performance.

## Project Goals

* Quantify and model market sentiment using Weibull-distributed decay functions.
* Integrate sentiment signals with technical indicators for price prediction.
* Develop and backtest a resilient and optimized low- to medium-frequency trading framework.

## Group Members

* [Leo Qi Jie Justin](https://github.com/Leo-QJ-Justin)
* [Justin Cheong](https://github.com/Justin-czk)
* [Ethan Cheung](https://github.com/ethan-cyj)
* [Chew Yu Cai](https://github.com/chewytry)

## Repository Structure

* `data/`: Contains exploratory data analysis (EDA) on news data.
* `data_preprocessing/`: Scripts and notebooks for preprocessing sentiment data.
* `notebooks/`: Contains notebooks used for analysis and strategy implementation.
* `report.pdf`: [View Project Report](https://github.com/ethan-cyj/DSE4211-Group-BTC-Day-Trading/blob/5ae124f57b9038287c79d3795ff8908b6e20fe1b/report.pdf)
* `TiktokenTime_Ethan(E0773997)_YuCai(E0726471)_JustinLeo(E0774569)_JustinCheong(E0773616).pdf`: [View Project Presentation Slides](https://github.com/ethan-cyj/DSE4211-Group-BTC-Day-Trading/blob/5ae124f57b9038287c79d3795ff8908b6e20fe1b/TiktokenTime_%20Ethan(E0773997)_YuCai(E0726471)_%20JustinLeo(E0774569)_JustinCheong(E0773616).pdf)

## Results

Our developed strategy significantly outperformed a traditional buy-and-hold approach, achieving a Sharpe Ratio of 7.18 compared to -0.52 for the benchmark, with an annualized return (CAGR) of 45.77% and reduced volatility.

## References

All relevant citations and detailed discussions are included in the [Project Report](https://github.com/ethan-cyj/DSE4211-Group-BTC-Day-Trading/blob/5ae124f57b9038287c79d3795ff8908b6e20fe1b/report.pdf).

---

**This project was conducted as part of the DSE4211 Digital Currencies course.**
