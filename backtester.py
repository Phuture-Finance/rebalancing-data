from pycoingecko import CoinGeckoAPI
import decouple
import pandas as pd

key = decouple.config("CG_KEY")
cg = CoinGeckoAPI(api_key=key)


class BackTester:
    def __init__(self, assets, weights) -> None:
        assert len(weights) == len(assets)
        self.assets = assets
        self.data = pd.DataFrame()
        self.weights = weights

    def get_data(self):
        data = pd.DataFrame()
        for i in self.assets:
            temp_data = pd.DataFrame(cg.get_coin_market_chart_by_id(i, "usd", "max"))
            new_df = pd.DataFrame(temp_data["prices"].to_list(),columns=['date',i])
            new_df.drop(new_df.tail(1).index,inplace=True)
            new_df['date'] = pd.to_datetime(new_df['date'],unit='ms')
            new_df.set_index('date',inplace=True)
            self.data = pd.concat([self.data,new_df],axis=1)
        common_date = None
        for column in self.data.columns:
            if common_date == None :
                common_date = self.data[column].first_valid_index()
            else:
                if self.data[column].first_valid_index() > common_date:
                    common_date = self.data.iloc[column].first_valid_index()
        self.data = self.data[self.data.index >= common_date]
        



            


bt = BackTester(["uniswap","compound-governance-token"],[0.1,0.2])
bt.get_data()
