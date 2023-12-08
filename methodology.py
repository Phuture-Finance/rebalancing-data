import pandas as pd
import numpy as np
import requests
import decouple
import sys

sys.path.append("../")
import time
from pycoingecko import CoinGeckoAPI

key = decouple.config("CG_KEY")
cg = CoinGeckoAPI(api_key=key)


class MethodologyBase:
    def __init__(
        self,
        min_mcap,
        min_weight,
        max_weight,
        max_int_for_weight,
        circ_supply_threshold,
        liveness_threshold,
        liquidity_consistency,
        max_slippage,
        cg_category=None,
    ):
        self.min_mcap = min_mcap
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.circ_supply_threshold = circ_supply_threshold
        self.liveness_threshold = liveness_threshold
        self.liquidity_consistency = liquidity_consistency
        self.max_slippage = max_slippage
        self.cg_category = cg_category
        self.max_int_for_weight = max_int_for_weight
        self.category_data = None
        self.all_coin_data = None
        self.mcap_data = None
        self.weights = None
        self.weights_converted = None
        self.slippage_data = None
        self.blockchains = {
            "ethereum": "ethereum",
            "avalanche": "avalanche-2",
            "binance-smart-chain": "binancecoin",
            "polygon-pos": "matic-network",
            "arbitrum-one": "ethereum",
            "arbitrum-nova": "ethereum",
            "fantom": "fantom",
            "optimistic-ethereum": "ethereum",
            "base": "ethereum",
        }
        # URLs for 0x
        self.url_0x = {
            "ethereum": "https://api.0x.org/swap/v1/quote",
            "polygon-pos": "https://polygon.api.0x.org/swap/v1/quote",
            "binance-smart-chain": "https://bsc.api.0x.org/swap/v1/quote",
            "optimistic-ethereum": "https://optimism.api.0x.org/swap/v1/quote",
            "fantom": "https://fantom.api.0x.org/swap/v1/quote",
            "avalanche": "https://avalanche.api.0x.org/swap/v1/quote",
            "arbitrum-nova": "https://arbitrum.api.0x.org/swap/v1/quote",
            "arbitrum-one": "https://arbitrum.api.0x.org/swap/v1/quote",
            "base": "https://base.api.0x.org/",
        }
        self.header = {"0x-api-key": decouple.config("ZEROEX_KEY")}
        self.stablecoin_by_blockchain_info = {
            "ethereum": {
                "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "decimals": 6,
            },
            "avalanche": {
                "address": "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",
                "decimals": 6,
            },
            "polygon-pos": {
                "address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                "decimals": 6,
            },
            "arbitrum-nova": {
                "address": "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
                "decimals": 6,
            },
            "arbitrum-one": {
                "address": "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
                "decimals": 6,
            },
            "optimistic-ethereum": {
                "address": "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
                "decimals": 6,
            },
            "fantom": {
                "address": "0x04068DA6C83AFCFA0e13ba15A6696662335D5B75",
                "decimals": 6,
            },
            "binance-smart-chain": {
                "address": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
                "decimals": 18,
            },
            "base": {
                "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                "decimals": 6,
            },
        }

    def get_category_data(self):
        if self.cg_category is None:
            self.category_data = pd.DataFrame(
                cg.get_coins_markets("usd", order="market_cap_desc", per_page=250)
            )
        else:
            self.category_data = pd.DataFrame(
                cg.get_coins_markets(
                    "usd",
                    category=self.cg_category,
                    order="market_cap_desc",
                    per_page=250,
                )
            )
        # Removing tokens with a market cap below the threshold
        self.category_data = self.category_data[
            self.category_data["market_cap"] >= self.min_mcap
        ]
        self.category_data.set_index("id", inplace=True)
        self.category_data = self.category_data[
            [
                "symbol",
                "name",
                "current_price",
                "market_cap",
                "market_cap_rank",
                "fully_diluted_valuation",
                "circulating_supply",
                "total_supply",
                "max_supply",
            ]
        ]
        return self.category_data

    def add_assets_to_category(self, ids):
        coin_data = pd.DataFrame(cg.get_coins_markets("usd", ids=ids))
        coin_data.set_index("id", inplace=True)
        coin_data = coin_data[
            [
                "symbol",
                "name",
                "current_price",
                "market_cap",
                "market_cap_rank",
                "fully_diluted_valuation",
                "circulating_supply",
                "total_supply",
                "max_supply",
            ]
        ]
        self.category_data = pd.concat([self.category_data, coin_data])
        self.category_data.sort_values(by=["market_cap"], inplace=True, ascending=False)
        return self.category_data

    def remove_assets_from_category(self, ids):
        self.category_data.drop(ids, inplace=True, errors="ignore")

    def replace_ids(self, ids, replacement_ids):
        for i in range(len(ids)):
            self.category_data.rename(index={ids[i]: replacement_ids[i]}, inplace=True)
        return self.category_data

    def get_all_coin_data(self):
        self.all_coin_data = pd.DataFrame(cg.get_coins_list(include_platform=True))
        self.all_coin_data.set_index("id", inplace=True)
        return self.all_coin_data

    def filter_and_merge_coin_data(self, single_chain=None, df_to_remove=None):
        self.all_coin_data.query("index in @self.category_data.index", inplace=True)
        if df_to_remove:
            for df in df_to_remove:
                self.all_coin_data.drop(df.index, inplace=True, errors="ignore")
        for id, data in self.all_coin_data.iterrows():
            to_remove = True
            if single_chain:
                if single_chain in list(data["platforms"].keys()):
                    to_remove = False
                    self.all_coin_data.at[id, "platforms"] = {
                        single_chain: data["platforms"][single_chain]
                    }
                if self.get_blockchain_by_native_asset(id) == single_chain:
                    to_remove = False
                if to_remove == True:
                    self.all_coin_data.drop(id, inplace=True)
            else:
                for blockchain in list(data["platforms"].keys()):
                    if blockchain in self.blockchains.keys():
                        to_remove = False
                if id in self.blockchains.values():
                    to_remove = False
                if to_remove == True:
                    self.all_coin_data.drop(id, inplace=True)
        self.category_data = self.category_data.join(
            self.all_coin_data["platforms"], how="inner", on="id"
        )
        return self.category_data

    def token_supply_check(self):
        supply_check = (
            self.category_data["circulating_supply"]
            / self.category_data["total_supply"]
            > self.circ_supply_threshold
        )
        self.category_data = self.category_data[supply_check]
        return self.category_data

    def asset_maturity_check(self):
        for id, _ in self.category_data.iterrows():
            cg_data = cg.get_coin_market_chart_by_id(id, vs_currency="USD", days="max")
            df_prices = pd.DataFrame(cg_data["prices"], columns=["date", id])
            df_prices = df_prices[df_prices[id] > 0]
            df_prices["date"] = pd.to_datetime(df_prices["date"], unit="ms").dt.date
            df_prices["date"] = pd.to_datetime(df_prices["date"])
            df_prices = df_prices.set_index("date", drop=True)
            df_prices = df_prices.loc[~df_prices.index.duplicated(keep="first")]

            if len(df_prices) < self.liveness_threshold:
                print(
                    f"Excluding {id}, pricing data available only for {len(df_prices)} < {self.liveness_threshold} days"
                )
                self.category_data.drop(id, inplace=True)
            else:
                df_mcaps = pd.DataFrame(cg_data["market_caps"], columns=["date", id])
                df_mcaps = df_mcaps[df_mcaps[id] > 0]
                df_mcaps["date"] = pd.to_datetime(df_mcaps["date"], unit="ms").dt.date
                df_mcaps["date"] = pd.to_datetime(df_mcaps["date"])
                df_mcaps = df_mcaps.set_index("date", drop=True)
                df_mcaps = df_mcaps.loc[~df_mcaps.index.duplicated(keep="first")]

                if len(df_mcaps) < self.liveness_threshold:
                    print(
                        f"Note: {id}, marketcap data available only for {len(df_mcaps)} < {self.liveness_threshold} days"
                    )
        return self.category_data

    def calculate_slippage(self, buy_token, blockchain):
        decimals = self.stablecoin_by_blockchain_info[blockchain]["decimals"]
        sell_token_id = "usd-coin"
        try:
            query = {
                "buyToken": buy_token,
                "sellToken": self.stablecoin_by_blockchain_info[blockchain]["address"],
                "sellAmount": int(
                    1e2 / cg.get_price(sell_token_id, "usd")[sell_token_id]["usd"]
                )
                * 10**decimals,
                "enableSlippageProtection": "true",
            }
            time.sleep(0.2)
            # spot price is calculated as a price for 100$ swap
            resp = requests.get(
                self.url_0x[blockchain], params=query, headers=self.header
            )
            swap = resp.json()
            spot_price = float(swap["price"])

            query["sellAmount"] = (
                int(1e5 / cg.get_price(sell_token_id, "usd")[sell_token_id]["usd"])
                * 10**decimals
            )
            time.sleep(0.2)
            resp = requests.get(
                self.url_0x[blockchain], params=query, headers=self.header
            )
            swap = resp.json()
            del_price = float(swap["price"])

            slippage = del_price / spot_price - 1

            return {
                "spot price": spot_price,
                "delivery price": del_price,
                "slippage": slippage,
                "blockchain": blockchain,
            }

        except KeyError:
            print(buy_token, blockchain)
            return None

    def get_blockchain_by_native_asset(self, coin_id):
        for blockchain, native_asset in self.blockchains.items():
            if coin_id == native_asset:
                return blockchain
        return None

    def assess_liquidity(self):
        slippages = []
        # Iterate over each row of the dataframe
        for id, coin_data in self.category_data.iterrows():
            slippage_dict = {"slippage": float("-inf")}
            # If there are no platforms listed it is likely a native asset so we use symbol instead of address for the buy token
            if len(coin_data["platforms"].keys()) == 0:
                slippage_dict = self.calculate_slippage(
                    coin_data["symbol"].upper(), self.get_blockchain_by_native_asset(id)
                )
                # If response is not None then we replace the current slippage dictionary with the return one
                if slippage_dict is not None:
                    slippage_dict["id"] = id
                    slippages.append(slippage_dict)
                else:
                    continue
            else:
                # Iterate over each blockchain the asset is listed on
                for blockchain in coin_data["platforms"].keys():
                    # Check that the blockchain is supported
                    if blockchain in self.blockchains.keys():
                        temp_slippage_dict = self.calculate_slippage(
                            coin_data["platforms"][blockchain], blockchain
                        )
                        # If response is not None and the return slippage is less negative than what is stored in slippage_dict then replace
                        if (
                            temp_slippage_dict is not None
                            and temp_slippage_dict["slippage"]
                            > slippage_dict["slippage"]
                        ):
                            temp_slippage_dict["id"] = id
                            slippage_dict = temp_slippage_dict

                        else:
                            continue
                    else:
                        continue
                # Check whether asset is native to a supported blockchain
                blockchain = self.get_blockchain_by_native_asset(id)
                if blockchain is not None:
                    temp_slippage_dict = self.calculate_slippage(
                        coin_data["symbol"], blockchain
                    )
                    # If return slippage is less negative than what is stored in slippage_dict then replace
                    if (
                        temp_slippage_dict is not None
                        and temp_slippage_dict["slippage"] > slippage_dict["slippage"]
                    ):
                        temp_slippage_dict["id"] = id
                        slippage_dict = temp_slippage_dict
                # If length of slippage_dict is greater than 1 this means there is a valid response to store
                if len(slippage_dict) > 1:
                    slippages.append(slippage_dict)
                # Else slippage_dict stores the default value and thus no valid response has been stored
                else:
                    continue
        slippage_pd = (
            pd.DataFrame(slippages)
            .set_index("id")
            .sort_values(by=["slippage"], ascending=False)
        )
        self.category_data = self.category_data.filter(
            slippage_pd[slippage_pd["slippage"] > self.max_slippage].index, axis=0
        )
        self.slippage_data = slippage_pd
        return (self.category_data, self.slippage_data)

    def check_redstone_price_feeds(self):
        redstone_base_url = (
            "https://api.redstone.finance/prices?provider=redstone&symbols="
        )
        symbols = list(self.category_data["symbol"].str.upper())
        for s in symbols:
            if s == symbols[-1]:
                redstone_base_url += f"{s}"
            else:
                redstone_base_url += f"{s},"
        symbol_zip = list(zip(self.category_data.index, symbols))
        request = requests.get(redstone_base_url).json()
        for id, symbol in symbol_zip:
            try:
                request[symbol]["value"]
            except KeyError:
                print(f"Dropping {id} because a price feed is unavailable")
                self.category_data.drop(id, inplace=True)
        return self.category_data

    def set_mcap_data(self):
        self.mcap_data = self.category_data["market_cap"]
        return self.mcap_data

    def calculate_weights(self):
        self.set_mcap_data()
        if self.max_weight < (1 / len(self.category_data)):
            self.max_weight = 1 / len(self.category_data)
        self.weights = self.mcap_data.div(self.mcap_data.sum())
        while (self.weights > self.max_weight).any(axis=None):
            self.weights[self.weights > self.max_weight] = self.max_weight
            remainder = 1 - self.weights.sum()
            if remainder == float(0):
                break
            smaller_weights = self.weights[self.weights < self.max_weight]
            self.weights = self.weights.add(
                smaller_weights.div(smaller_weights.sum()).mul(remainder), fill_value=0
            )
        acceptable_weights = self.weights > self.min_weight
        self.weights = self.weights[acceptable_weights]
        self.weights = self.weights.div(self.weights.sum())
        self.weights.sort_values(inplace=True, ascending=False)
        self.category_data.query("index in @self.weights.index", inplace=True)
        self.set_mcap_data()
        return self.weights

    def converted_weights(self):
        w_scaled = self.weights * self.max_int_for_weight
        w_res = np.floor(w_scaled).astype(int)
        remainders = w_scaled - w_res
        k = round(remainders.sum())
        while k > 0:
            for i in w_res.index:
                if k > 0:
                    w_res[i] += 1
                    k -= 1
                else:
                    break
        self.weights_converted = w_res
        self.weights_converted.sort_values(inplace=True, ascending=False)
        assert (
            self.weights_converted.sum() == self.max_int_for_weight
        ), "Sum of converted weights does equal max int value"
        return self.weights_converted

    def show_results(self):
        results = pd.DataFrame()
        results.index = self.category_data.index
        results["name"] = self.category_data["name"]
        results["market_cap"] = self.category_data["market_cap"]
        results["weight"] = self.weights
        results["weight_converted"] = self.weights_converted
        results["address"] = [
            data["platforms"][self.slippage_data.at[id, "blockchain"]]
            if self.slippage_data.at[id, "blockchain"] in data["platforms"].keys()
            else data["symbol"].upper()
            for id, data in self.category_data.iterrows()
        ]
        results["blockchain_with_highest_liq"] = [
            self.slippage_data.at[id, "blockchain"]
            for id, data in self.category_data.iterrows()
        ]
        results = results.sort_values("market_cap", ascending=False)
        return results

    def main(
        self,
        single_chain=None,
        df_to_remove=None,
        add_category_assets=None,
        remove_category_assets=None,
        ids_to_replace=None,
    ):
        self.get_category_data()
        if remove_category_assets:
            self.remove_assets_from_category(remove_category_assets)
        if add_category_assets:
            self.add_assets_to_category(add_category_assets)
        if ids_to_replace:
            self.replace_ids(ids_to_replace[0], ids_to_replace[1])
        self.get_all_coin_data()
        self.filter_and_merge_coin_data(single_chain, df_to_remove)
        self.token_supply_check()
        self.asset_maturity_check()
        self.assess_liquidity()
        self.check_redstone_price_feeds()
        self.calculate_weights()
        self.converted_weights()
        return (self.show_results(), self.slippage_data)


class MethodologyProd(MethodologyBase):
    def __init__(
        self,
        index_homechain,
        index_address,
        min_mcap,
        min_weight,
        max_weight,
        max_int_for_weight,
        circ_supply_threshold,
        liveness_threshold,
        liquidity_consistency,
        max_slippage,
        cg_category=None,
    ):
        super().__init__(
            min_mcap,
            min_weight,
            max_weight,
            max_int_for_weight,
            circ_supply_threshold,
            liveness_threshold,
            liquidity_consistency,
            max_slippage,
            cg_category,
        )
        self.index_address = index_address
        self.index_homechain = index_homechain

    def main(
        self,
        single_chain=None,
        df_to_remove=None,
        add_category_assets=None,
        remove_category_assets=None,
        ids_to_replace=None,
    ):
        super().main(
            single_chain,
            df_to_remove,
            add_category_assets,
            remove_category_assets,
            ids_to_replace,
        )

        
