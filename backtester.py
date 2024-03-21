from pycoingecko import CoinGeckoAPI
import decouple
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


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
        assert type(assets) == dict
        assert type(benchmark_assets) == list
        self.assets = assets
        self.benchmark_assets = benchmark_assets
        self.raw_data = pd.DataFrame()
        self.return_data = pd.DataFrame()

        sum = 0
        for i in self.assets.values():
            sum += i
        assert round(sum, 2) == 1.0
        self.get_data()

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
        self.raw_data = self.raw_data.pct_change()
        self.raw_data.dropna(axis=0, inplace=True)

    def add_benchmarks_to_data(self, dataframe):
        temp_data = pd.DataFrame()
        for asset in self.benchmark_assets:
            print(asset)
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
            asset_prices[f"{asset}-daily-rtn"].fillna(0, limit=1, inplace=True)
            asset_prices["adj-rtn"] = asset_prices[f"{asset}-daily-rtn"] + 1
            asset_prices[f"{asset}-cum-rtn"] = (
                asset_prices["adj-rtn"].cumprod(axis=0).subtract(1)
            )
            asset_prices[f"{asset}-cum-rtn"].fillna(1, limit=1, inplace=True)
            temp_data = pd.concat([temp_data, asset_prices[f"{asset}-cum-rtn"]], axis=1)
        dataframe = pd.concat([dataframe, temp_data], axis=1)
        return dataframe

    def calculate_correlation(self,dataframe):
        daily_returns = dataframe.filter(like="daily")
        daily_returns = self.add_benchmarks_to_data(dataframe)
        return daily_returns.corr(method="pearson")

    def calculate_sharpe_ratio(self, return_data,risk_free_rate):
        mean_return = return_data.mean()
        standard_dev = return_data.std()
        return (mean_return - risk_free_rate / 365) / standard_dev 

    def simulate_with_rebalancing(self, days, liquidity_discount):
        initial_value = 1000000
        composition = {}
        for i in self.assets.keys():
            composition[i] = [initial_value * self.assets[i]]
        composition["date"] = [
            pd.to_datetime(
                self.raw_data.first_valid_index().timestamp() - 86400, unit="s"
            )
        ]
        composition[f"Index value w/ {days} days rebalancing window"] = [initial_value]
        count = 0
        rebalancing_dates = []
        for index, row in self.raw_data.iterrows():
            composition["date"].append(index)
            if count % days == 0:
                rebalancing_dates.append(index)
                current_value = composition[
                    f"Index value w/ {days} days rebalancing window"
                ][-1]
                total = 0
                for i in self.assets.keys():
                    value = (
                        current_value
                        * self.assets[i]
                        * (1 + row[i])
                        * (1 - liquidity_discount)
                    )
                    composition[i].append(value)
                    total += value
                composition[f"Index value w/ {days} days rebalancing window"].append(
                    total
                )
            else:
                total = 0
                for i in self.assets:
                    value = composition[i][-1] * (1 + row[i])
                    composition[i].append(value)
                    total += value
                composition[f"Index value w/ {days} days rebalancing window"].append(
                    total
                )
            count += 1

        df = pd.DataFrame(composition)
        df["date"] = pd.to_datetime(df["date"], unit="s")
        df.set_index("date", inplace=True)

        return_df = pd.DataFrame()
        return_df[f"Index cum-return w/ {days} days rebalancing window"] = (
            df[f"Index value w/ {days} days rebalancing window"]
            .div(initial_value)
            .subtract(1)
        )
        return_df[f"Index daily-return w/ {days} days rebalancing window"] = df[f"Index value w/ {days} days rebalancing window"].pct_change()

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Change in Index Portfolio Value", "Index Rate of Return"),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1,
        )
        for i in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[i], mode="lines", name=i), row=1, col=1
            )

        fig.add_trace(
            go.Scatter(x=return_df.index, y=return_df[f"Index cum-return w/ {days} days rebalancing window"], mode="lines", name="Return"),
            row=2,
            col=1,
        )
        for date in rebalancing_dates:
            fig.add_vline(
                x=date,
                line_dash="dot",
                line_color="green",
                line_width=1,
                label={"text": "Rebalancing", "textposition": "middle right"},
            )
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Investment Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=2, col=1)
        fig.update_layout(title="Index Performance Metrics", height=1000, width=1100)

        return fig, df, return_df

    def assess_different_rebalancing_periods(self, periods, liquidity_discount):
        concat_data = pd.DataFrame()
        for i in periods:
            _, _, return_data = self.simulate_with_rebalancing(i, liquidity_discount)
            return_data = return_data.filter(like='cum')
            concat_data = pd.concat([concat_data, return_data], axis=1)
        concat_data = self.add_benchmarks_to_data(concat_data)
        fig = px.line(concat_data)
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Return (%)")
        fig.update_layout(title="Index Performance")

        return concat_data, fig
