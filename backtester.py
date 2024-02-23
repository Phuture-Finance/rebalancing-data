from pycoingecko import CoinGeckoAPI
import decouple
import pandas as pd
import numpy as np
import plotly.express as px

key = decouple.config("CG_KEY")
cg = CoinGeckoAPI(api_key=key)


def convert_results_table(results):
    assets = list(results.index)
    weights = list(results["weight"])
    output_dict = {}
    for i in range(len(assets)):
        output_dict[assets[i]] = weights[i]

    return output_dict

class BackTester:
    def __init__(self, assets, benchmark_assets) -> None:
        self.assets = assets
        self.benchmark_assets = benchmark_assets
        self.raw_data = pd.DataFrame()
        self.return_data = pd.DataFrame()

    def get_data(self):
        for i in self.assets.keys():
            data = pd.DataFrame(cg.get_coin_market_chart_by_id(i, "usd", "max"))
            data = pd.DataFrame(data["prices"].to_list(), columns=["date", i])
            data.drop(data.tail(1).index, inplace=True)
            data["date"] = pd.to_datetime(data["date"], unit="ms")
            data.set_index("date", inplace=True)
            self.raw_data = pd.concat([self.raw_data, data], axis=1)
        common_date = None
        for column in self.raw_data.columns:
            if common_date == None:
                common_date = self.raw_data[column].first_valid_index()
            else:
                if self.raw_data[column].first_valid_index() > common_date:
                    common_date = self.raw_data[column].first_valid_index()
        self.raw_data = self.raw_data[self.raw_data.index >= common_date]
        self.raw_data.dropna(axis=0, inplace=True)
        self.raw_data = self.raw_data.pct_change()

        self.return_data["index-daily-rtn"] = self.raw_data.apply(
            self.calculate_index_return, axis=1
        )
        self.return_data["index-daily-rtn"].fillna(0,limit=1,inplace=True)
        self.return_data["adj-rtn"] = self.return_data["index-daily-rtn"] + 1
        self.return_data["index-cum-rtn"] = self.return_data["adj-rtn"].cumprod(axis=0)
        self.return_data["index-cum-rtn"].fillna(1,limit=1,inplace=True)
        self.return_data.drop(columns=["adj-rtn"], inplace=True)

    def calculate_index_return(self, row):
        rtn = 0
        for i in row.index:
            weight = self.assets[i]
            rtn += weight * row[i]
        return rtn

    def add_benchmarks_to_data(self):
        for asset in self.benchmark_assets:
            asset_prices = pd.DataFrame(
                cg.get_coin_market_chart_range_by_id(
                    asset,
                    "usd",
                    self.raw_data.index[0].timestamp(),
                    self.raw_data.last_valid_index().timestamp(),
                )
            )
            asset_prices = pd.DataFrame(
                asset_prices["prices"].to_list(), columns=["date", "price"]
            )
            asset_prices["date"] = pd.to_datetime(asset_prices["date"], unit="ms")
            asset_prices.set_index("date", inplace=True)
            asset_prices[f"{asset}-daily-rtn"] = asset_prices["price"].pct_change()
            asset_prices[f"{asset}-daily-rtn"].fillna(0,limit=1,inplace=True)
            asset_prices["adj-rtn"] = asset_prices[f"{asset}-daily-rtn"] + 1
            asset_prices[f"{asset}-cum-rtn"] = asset_prices["adj-rtn"].cumprod(axis=0)
            asset_prices[f"{asset}-cum-rtn"].fillna(1,limit=1,inplace=True)
            asset_prices.drop(columns=["price", "adj-rtn"], inplace=True)
            self.return_data = pd.concat([self.return_data, asset_prices], axis=1)

    def calculate_correlation(self):
        daily_returns = self.return_data.filter(like='daily-rtn')
        return daily_returns.corr(method='pearson')
    
    def calculate_sharpe_ratio(self,risk_free_rate):
        mean_return = self.return_data['index-daily-rtn'].mean()
        standard_dev = self.return_data['index-daily-rtn'].std()
        return (mean_return - risk_free_rate/365)/standard_dev
    
    def graph_returns(self):
        returns = self.return_data.filter(like='cum-rtn')
        return px.line(returns,labels={'date':'Date','value': 'Cumulative Return','index-cum-rtn':'Index'},title='Performance of Proposed Index',)


    def main(self):
        self.get_data()
        self.add_benchmarks_to_data()
        correlations = self.calculate_correlation()
        sharpe_ratio = self.calculate_sharpe_ratio()
        return self.return_data,correlations,sharpe_ratio
