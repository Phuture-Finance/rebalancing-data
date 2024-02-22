from pycoingecko import CoinGeckoAPI
import decouple
import pandas as pd
import numpy as np

key = decouple.config("CG_KEY")
cg = CoinGeckoAPI(api_key=key)


class BackTester:
    def __init__(self, assets) -> None:
        self.assets = assets
        self.data = pd.DataFrame()
        self.btc = None
        self.eth = None

    def get_data(self):
        for i in self.assets.keys():
            temp_data = pd.DataFrame(cg.get_coin_market_chart_by_id(i, "usd", "max"))
            new_df = pd.DataFrame(temp_data["prices"].to_list(), columns=["date", i])
            new_df.drop(new_df.tail(1).index, inplace=True)
            new_df["date"] = pd.to_datetime(new_df["date"], unit="ms")
            new_df.set_index("date", inplace=True)
            self.data = pd.concat([self.data, new_df], axis=1)
        common_date = None
        for column in self.data.columns:
            if common_date == None:
                common_date = self.data[column].first_valid_index()
            else:
                if self.data[column].first_valid_index() > common_date:
                    common_date = self.data[column].first_valid_index()
        self.data = self.data[self.data.index >= common_date]
        self.data.dropna(axis=0, inplace=True)
        self.data = self.data.pct_change()
        self.data.dropna(axis=0, inplace=True)
        self.data["index"] = self.data.apply(self.calculate_index_return, axis=1)

    def calculate_index_return(self, row):
        rtn = 0
        for i in row.index:
            weight = self.assets[i]
            rtn += weight * row[i]
        return rtn
    
    def cumualative_performance(self):
        price = 100
        for index, row in self.data.iterrows():
            price *= 1+row['index']
        return price

bt = BackTester(
    {
        "uniswap": 0.3,
        "compound-governance-token": 0.2,
        "rocket-pool": 0.2,
        "gmx": 0.1,
        "pancakeswap-token": 0.1,
        "aave": 0.1,
    }
)
bt.get_data()
print(bt.cumualative_performance())
