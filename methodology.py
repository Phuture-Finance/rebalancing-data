import pandas as pd
import numpy as np
import requests
import decouple
import sys
from web3 import Web3
import datetime

sys.path.append("../")
import time
from pycoingecko import CoinGeckoAPI
from abis import index_anatomy
from db_funcs import (
    convert_to_sql_strings,
    insert_values,
    create_connection,
    db,
    convert_from_sql_strings,
    create_benchmark_table,
    create_liquidity_table,
)

key = decouple.config("CG_KEY")
cg = CoinGeckoAPI(api_key=key)


class MethodologyBase:
    def __init__(
        self,
        index_homechain,
        min_mcap,
        min_weight,
        max_weight,
        max_int_for_weight,
        circ_supply_threshold,
        liveness_threshold,
        liquidity_consistency,
        max_slippage,
        cg_categories=None,
    ):
        self.index_homechain = index_homechain

        self.min_mcap = min_mcap
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.circ_supply_threshold = circ_supply_threshold
        self.liveness_threshold = liveness_threshold
        self.liquidity_consistency = liquidity_consistency
        self.max_slippage = max_slippage
        self.cg_categories = cg_categories
        self.max_int_for_weight = max_int_for_weight
        self.category_data = None
        self.all_coin_data = None
        self.mcap_data = None
        self.weights = None
        self.weights_converted = None
        self.slippage_data = None
        self.results = None
        self.blockchains = {
            "ethereum": "ethereum",
            "avalanche": "avalanche-2",
            "binance-smart-chain": "binancecoin",
            "polygon-pos": "matic-network",
            "arbitrum-one": "ethereum",
            "fantom": "fantom",
            "optimistic-ethereum": "ethereum",
            "base": "ethereum",
        }
        # URLs for 0x
        self.url_0x = {
            "ethereum": "https://api.0x.org/swap/v1/price",
            "polygon-pos": "https://polygon.api.0x.org/swap/v1/price",
            "binance-smart-chain": "https://bsc.api.0x.org/swap/v1/price",
            "optimistic-ethereum": "https://optimism.api.0x.org/swap/v1/price",
            "fantom": "https://fantom.api.0x.org/swap/v1/price",
            "avalanche": "https://avalanche.api.0x.org/swap/v1/price",
            "arbitrum-one": "https://arbitrum.api.0x.org/swap/v1/price",
            "base": "https://base.api.0x.org/swap/v1/price",
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
        assert self.index_homechain in list(
            self.blockchains.keys()
        ), "Homechain not supported"
        assert type(self.cg_categories) == list, "Categories must be in a list"

    def get_category_data(self):
        if self.cg_categories is None:
            self.category_data = pd.DataFrame(
                cg.get_coins_markets("usd", order="market_cap_desc", per_page=250)
            )
        else:
            dataframe_list = []
            for category in self.cg_categories:
                dataframe_list.append(
                    pd.DataFrame(
                        cg.get_coins_markets(
                            "usd",
                            category=category,
                            order="market_cap_desc",
                            per_page=250,
                        )
                    )
                )
            self.category_data = pd.concat(dataframe_list, axis=0)
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
            ]
        ]
        self.category_data["symbol"] = self.category_data["symbol"].str.upper()
        self.category_data.drop_duplicates(inplace=True)
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
            ]
        ]
        coin_data["symbol"] = coin_data["symbol"].str.upper()
        self.category_data = pd.concat([self.category_data, coin_data])
        self.category_data.drop_duplicates(inplace=True)
        self.category_data.sort_values(by=["market_cap"], inplace=True, ascending=False)
        return self.category_data

    def remove_assets_from_category(self, ids):
        self.category_data.drop(
            ids,
            inplace=True,
            errors="ignore",
        )

    def replace_ids(self, ids, replacement_ids):
        for i in range(len(ids)):
            self.category_data.rename(index={ids[i]: replacement_ids[i]}, inplace=True)
        return self.category_data

    def update_token_data(self, data):
        for dictionary in data:
            self.category_data.at[dictionary["id"], dictionary["category"]] = (
                dictionary["value"]
            )

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

    def add_platform_to_token(self, data):
        for dictionary in data:
            self.category_data.at[dictionary["id"], "platforms"][
                dictionary["blockchain"]
            ] = dictionary["address"]

    def token_supply_check(self):
        supply_check = (
            self.category_data["circulating_supply"]
            / self.category_data["total_supply"]
            >= self.circ_supply_threshold
        )
        self.category_data = self.category_data[supply_check]
        return self.category_data

    def asset_maturity_check(self):
        for id, _ in self.category_data.iterrows():
            cg_data = cg.get_coin_market_chart_by_id(id, vs_currency="usd", days="max")
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
        return self.category_data

    def calculate_slippage(self, id, buy_token, blockchain):
        stable_coin_decimals = self.stablecoin_by_blockchain_info[blockchain][
            "decimals"
        ]
        stable_coin_id = "usd-coin"
        try:
            buy_query = {
                "buyToken": buy_token,
                "sellToken": self.stablecoin_by_blockchain_info[blockchain]["address"],
                "sellAmount": int(
                    1e5
                    / cg.get_price(stable_coin_id, "usd")[stable_coin_id]["usd"]
                    * 10**stable_coin_decimals
                ),
                "enableSlippageProtection": "true",
            }

            sell_query = {
                "buyToken": self.stablecoin_by_blockchain_info[blockchain]["address"],
                "sellToken": buy_token,
                "sellAmount": int(1e5 / cg.get_price(id, "usd")[id]["usd"] * 10**18),
                "enableSlippageProtection": "true",
            }
            buy_swap = requests.get(
                self.url_0x[blockchain], params=buy_query, headers=self.header
            ).json()

            sell_swap = requests.get(
                self.url_0x[blockchain], params=sell_query, headers=self.header
            ).json()
            if (
                buy_swap["estimatedPriceImpact"] != None
                and sell_swap["estimatedPriceImpact"] != None
            ):
                slippage = min(
                    -float(buy_swap["estimatedPriceImpact"]) / 100,
                    -float(sell_swap["estimatedPriceImpact"]) / 100,
                )
                return [blockchain, slippage]
            else:
                # If estimated slippage field on API response is none we manually check the slippage by comparing the slippage on a large swap to a small one
                print(f"Manually calculating slippage for {id} on {blockchain}")
                buy_query["sellAmount"] = int(
                    10
                    / cg.get_price(stable_coin_id, "usd")[stable_coin_id]["usd"]
                    * 10**stable_coin_decimals
                )
                sell_query["sellAmount"] = int(
                    10 / cg.get_price(id, "usd")[id]["usd"] * 10**18
                )

                small_buy_swap = requests.get(
                    self.url_0x[blockchain], params=buy_query, headers=self.header
                ).json()

                small_sell_swap = requests.get(
                    self.url_0x[blockchain], params=sell_query, headers=self.header
                ).json()
                slippage = min(
                    float(buy_swap["price"]) / float(small_buy_swap["price"]) - 1,
                    float(sell_swap["price"]) / float(small_sell_swap["price"]) - 1,
                )
                return [blockchain, slippage]

        except KeyError:
            return None

    def get_blockchain_by_native_asset(self, coin_id):
        for blockchain, native_asset in self.blockchains.items():
            if coin_id == native_asset:
                return blockchain
        return None

    def assess_liquidity(self, blockchains_to_remove=None):
        slippages = pd.DataFrame()
        # Iterate over each row of the dataframe
        for id, coin_data in self.category_data.iterrows():
            slippage_dict = {}
            # If there are no platforms listed it is likely a native asset so we use symbol instead of address for the buy token
            if len(coin_data["platforms"].keys()) == 0:
                slippage_check = self.calculate_slippage(
                    id,
                    coin_data["symbol"].upper(),
                    self.get_blockchain_by_native_asset(id),
                )
                if slippage_check is not None:
                    slippage_dict["id"] = [id]
                    slippage_dict[slippage_check[0]] = [slippage_check[1]]
                else:
                    print(
                        f"{id} with no additional platforms returned an invalid API response"
                    )
                    continue
            else:
                # Iterate over each blockchain the asset is listed on
                for blockchain in coin_data["platforms"].keys():
                    # Check that the blockchain is supported
                    if blockchain in self.blockchains.keys():
                        slippage_check = self.calculate_slippage(
                            id, coin_data["platforms"][blockchain], blockchain
                        )
                        if slippage_check is not None:
                            slippage_dict["id"] = [id]
                            slippage_dict[slippage_check[0]] = [slippage_check[1]]

                        else:
                            print(
                                f"{id} on {blockchain} returned an invalid API response"
                            )
                            continue
                    else:
                        print(f"{blockchain} not supported")
                        continue
                # Check whether asset is native to a supported blockchain
                blockchain = self.get_blockchain_by_native_asset(id)
                if blockchain is not None:
                    slippage_check = self.calculate_slippage(
                        id, coin_data["symbol"], blockchain
                    )
                    if slippage_check is not None:
                        slippage_dict["id"] = [id]
                        slippage_dict[slippage_check[0]] = [slippage_check[1]]
            # If length of slippage_dict is greater than 0 this means there is a valid response to store
            if len(slippage_dict) > 0:
                temp_dataframe = pd.DataFrame(slippage_dict)
                temp_dataframe.set_index("id", inplace=True)
                slippages = pd.concat([slippages, temp_dataframe], axis=1)
            else:
                print(f"{id} recorded no valid response across any platform")
                continue
        slippages = slippages.T.groupby(level=0).sum().T
        slippages.replace(0, np.nan, inplace=True)
        if blockchains_to_remove:
            slippages.drop(columns=blockchains_to_remove,inplace=True,errors='ignore')
        acceptable_slippages = slippages[slippages >= self.max_slippage].dropna(
            axis=0, how="all"
        )
        self.category_data = self.category_data.filter(
            acceptable_slippages.index, axis=0
        )
        slippages["optimal chain"] = slippages.apply(
            self.compute_chain_for_optimal_liquidity, axis=1
        )
        slippages["best slippage"] = slippages.max(
            axis=1, skipna=True, numeric_only=True
        )
        slippages["best slippage chain"] = slippages.idxmax(
            axis=1, skipna=True, numeric_only=True
        )
        slippages.sort_values(
            by="best slippage",
            ascending=False,
            inplace=True,
        )

        self.slippage_data = slippages
        return (self.category_data, self.slippage_data)

    def compute_chain_for_optimal_liquidity(self, series):
        filtered_series = series[series > self.max_slippage]
        if len(filtered_series.index) == 0:
            return "None"
        elif len(filtered_series.index) == 1:
            return filtered_series.index[0]
        elif self.index_homechain in filtered_series.index:
            return self.index_homechain
        elif "ethereum" in filtered_series.index:
            filtered_series.drop(labels=["ethereum"], inplace=True)
            return filtered_series.idxmax()
        else:
            return filtered_series.idxmax()

    def check_redstone_price_feeds(self, onchain_oracles):
        redstone_base_url = (
            "https://api.redstone.finance/prices?provider=redstone&symbols="
        )
        symbols = list(self.category_data["symbol"])
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
                if onchain_oracles is not None and id in onchain_oracles:
                    continue
                else:
                    print(f"Dropping {id} because a price feed is unavailable")
                    self.category_data.drop(id, inplace=True)
        return self.category_data

    def set_mcap_data(self):
        self.mcap_data = self.category_data["market_cap"]
        return self.mcap_data

    def calculate_weights(self, split_data=None):
        self.set_mcap_data()
        if self.max_weight < (1 / len(self.category_data)):
            self.max_weight = 1 / len(self.category_data)
        if (
            split_data != None
            and split_data["asset_to_split"] in self.mcap_data.index
            and split_data["asset_to_receive"] in self.mcap_data.index
        ):
            temp_mcap_data = self.mcap_data
            temp_mcap_data.drop(split_data["asset_to_receive"], inplace=True)
            self.weights = temp_mcap_data.div(temp_mcap_data.sum())
        else:
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
        acceptable_weights = self.weights >= self.min_weight
        unacceptable_weights = self.weights[self.weights < self.min_weight]
        for asset in unacceptable_weights.index:
            print(f"{asset} does not meet the minimum weight requirement")
        self.weights = self.weights[acceptable_weights]
        self.weights = self.weights.div(self.weights.sum())
        if split_data != None:
            self.weights[split_data["asset_to_receive"]] = (
                self.weights[split_data["asset_to_split"]] * split_data["split_ratio"]
            )
            self.weights[split_data["asset_to_split"]] = self.weights[
                split_data["asset_to_split"]
            ] * (1 - split_data["split_ratio"])
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
        results["symbol"] = self.category_data["symbol"]
        results["market_cap"] = self.category_data["market_cap"]
        results["weight"] = self.weights
        results["weight_converted"] = self.weights_converted

        address_list = []
        blockchain_list = []

        for id, data in self.category_data.iterrows():
            chain = self.slippage_data.at[id, "optimal chain"]
            blockchain_list.append(chain)
            if chain in data["platforms"].keys():
                address_list.append(data["platforms"][chain])
            else:
                address_list.append("0x0000000000000000000000000000000000000000")
        results["address"] = address_list
        results["blockchain"] = blockchain_list
        results.sort_values(by=["market_cap"], ascending=False, inplace=True)
        self.results = results
        return self.results

    def main(
        self,
        single_chain=None,
        df_to_remove=None,
        add_category_assets=None,
        remove_category_assets=None,
        ids_to_replace=None,
        values_to_update=None,
        platforms_to_add=None,
        platforms_to_remove=None,
        weight_split_data=None,
        onchain_oracles=None,
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
        if platforms_to_add:
            self.add_platform_to_token(platforms_to_add)        
        if values_to_update:
            self.update_token_data(values_to_update)
        self.token_supply_check()
        self.asset_maturity_check()
        self.assess_liquidity(platforms_to_remove)
        self.check_redstone_price_feeds(onchain_oracles)
        self.calculate_weights(weight_split_data)
        self.converted_weights()
        return (self.show_results(), self.slippage_data)


class MethodologyProd(MethodologyBase):
    def __init__(
        self,
        date,
        version,
        index_homechain,
        index_address,
        db_benchmark_table,
        db_liquidity_table,
        min_mcap,
        min_weight,
        max_weight,
        max_int_for_weight,
        circ_supply_threshold,
        liveness_threshold,
        liquidity_consistency,
        max_slippage,
        cg_category=None,
        subgraph_url=None,
    ):
        super().__init__(
            index_homechain,
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
        self.date = date
        self.version = version
        self.index_address = index_address
        self.w3 = Web3(
            Web3.HTTPProvider(self.chain_to_provider_url(self.index_homechain))
        )
        self.db_benchmark_table = db_benchmark_table
        self.db_liquidity_table = db_liquidity_table
        self.avg_slippage_data = None
        self.subgraph_url = subgraph_url

    def main(
        self,
        single_chain=None,
        df_to_remove=None,
        add_category_assets=None,
        remove_category_assets=None,
        ids_to_replace=None,
        values_to_update=None,
        platforms_to_add=None,
        platforms_to_remove=None,
        weight_split_data=None,
        onchain_oracles=None,
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
        if platforms_to_add:
            self.add_platform_to_token(platforms_to_add)
        if values_to_update:
            self.update_token_data(values_to_update)
        self.token_supply_check()
        self.asset_maturity_check()
        self.create_db_tables()
        self.assess_liquidity(platforms_to_remove)
        self.check_redstone_price_feeds(onchain_oracles)
        self.calculate_weights(weight_split_data)
        self.converted_weights()
        self.show_results()

        self.add_results_to_benchmark_db()
        if self.version == 1:
            self.v1_index_diff_check()
        return (self.results, self.slippage_data)

    def chain_to_provider_url(self, chain):
        mapping = {
            "ethereum": decouple.config("ETHEREUM_INFURA_URL"),
            "avalanche": decouple.config("AVALANCHE_INFURA_URL"),
            "binance-smart-chain": decouple.config("BINANCE_INFURA_URL"),
            "polygon-pos": decouple.config("POLYGON_INFURA_URL"),
            "arbitrum-one": decouple.config("ARBITRUM_INFURA_URL"),
            "optimistic-ethereum": decouple.Config("OPTIMISM_INFURA_URL"),
            "base": decouple.config("BASE_INFURA_URL"),
        }

        return mapping[chain]

    def chain_to_chain_id(self, chain):
        mapping = {
            "ethereum": 1,
            "avalanche": 43114,
            "binance-smart-chain": 56,
            "polygon-pos": 137,
            "arbitrum-one": 42161,
            "optimistic-ethereum": 10,
            "base": 8453,
        }

        return mapping[chain]

    def v1_index_diff_check(self):
        anatomy_contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(self.index_address), abi=index_anatomy
        )
        addresses, weights = anatomy_contract.functions.anatomy().call()
        for address in addresses:
            address = address.lower()
            if address not in list(self.results["address"]):
                data = cg.get_coin_info_from_contract_address_by_id(
                    self.index_homechain, address
                )
                self.results.loc[data["id"]] = [data["name"], 0, 0, 0, address, "None"]
        return self.results

    def add_results_to_benchmark_db(self):
        db_assets = convert_to_sql_strings(list(self.results.index))
        db_weights = list(self.results["weight"])
        insert_values(self.date, db_assets, db_weights, self.db_benchmark_table)

    def add_slippage_to_liquidity_db(self):
        slippages_to_save = self.slippage_data["best slippage"].head(20)
        print(slippages_to_save)
        asset_columns = convert_to_sql_strings(list(slippages_to_save.index))
        print(asset_columns)
        slippage_values = list(slippages_to_save.values)
        insert_values(
            self.date, asset_columns, slippage_values, self.db_liquidity_table
        )

    def output_for_contract(self):
        if self.version == 1:
            zipped_assets = list(
                zip(
                    list(self.results["address"]),
                    list(self.results["weight_converted"]),
                )
            )
            sorted_assets = sorted(zipped_assets, key=lambda x: int(x[0], base=0))
            asset_string = []
            weight_string = []
            for asset, weight in sorted_assets:
                asset_string.append(asset)
                weight_string.append(f"{weight}")
            asset_string = ",".join(asset_string)
            weight_string = ",".join(weight_string)
            print(asset_string)
            print(weight_string)

    def assess_liquidity(self,blockchains_to_remove):
        slippages = pd.DataFrame()
        # Iterate over each row of the dataframe
        for id, coin_data in self.category_data.iterrows():
            slippage_dict = {}
            # If there are no platforms listed it is likely a native asset so we use symbol instead of address for the buy token
            if len(coin_data["platforms"].keys()) == 0:
                slippage_check = self.calculate_slippage(
                    id,
                    coin_data["symbol"].upper(),
                    self.get_blockchain_by_native_asset(id),
                )
                if slippage_check is not None:
                    slippage_dict["id"] = [id]
                    slippage_dict[slippage_check[0]] = [slippage_check[1]]
                else:
                    print(
                        f"{id} with no additional platforms returned an invalid API response"
                    )
                    continue
            else:
                # Iterate over each blockchain the asset is listed on
                for blockchain in coin_data["platforms"].keys():
                    # Check that the blockchain is supported
                    if blockchain in self.blockchains.keys():
                        slippage_check = self.calculate_slippage(
                            id, coin_data["platforms"][blockchain], blockchain
                        )
                        if slippage_check is not None:
                            slippage_dict["id"] = [id]
                            slippage_dict[slippage_check[0]] = [slippage_check[1]]

                        else:
                            print(
                                f"{id} on {blockchain} returned an invalid API response"
                            )
                            continue
                    else:
                        print(f"{blockchain} not supported")
                        continue
                # Check whether asset is native to a supported blockchain
                blockchain = self.get_blockchain_by_native_asset(id)
                if blockchain is not None:
                    slippage_check = self.calculate_slippage(
                        id, coin_data["symbol"], blockchain
                    )
                    if slippage_check is not None:
                        slippage_dict["id"] = [id]
                        slippage_dict[slippage_check[0]] = [slippage_check[1]]
            # If length of slippage_dict is greater than 0 this means there is a valid response to store
            if len(slippage_dict) > 0:
                temp_dataframe = pd.DataFrame(slippage_dict)
                temp_dataframe.set_index("id", inplace=True)
                slippages = pd.concat([slippages, temp_dataframe], axis=1)
            else:
                print(f"{id} recorded no valid response across any platform")
                continue
        slippages = slippages.T.groupby(level=0).sum().T
        slippages.replace(0, np.nan, inplace=True)
        if blockchains_to_remove:
            slippages.drop(columns=blockchains_to_remove,inplace=True,errors='ignore')
            slippages.dropna(axis=0,how='all',inplace=True)
        slippages["optimal chain"] = slippages.apply(
            self.compute_chain_for_optimal_liquidity, axis=1
        )
        slippages["best slippage"] = slippages.max(
            axis=1, skipna=True, numeric_only=True
        )
        slippages["best slippage chain"] = slippages.idxmax(
            axis=1, skipna=True, numeric_only=True
        )
        slippages.sort_values(
            by="best slippage",
            ascending=False,
            inplace=True,
        )
        self.slippage_data = slippages
        self.add_slippage_to_liquidity_db()

        self.avg_slippage_data = self.check_avg_slippage()
        self.category_data = self.category_data.filter(
            self.avg_slippage_data.index, axis=0
        )

        return (self.category_data, self.slippage_data)

    def create_db_tables(self):
        create_liquidity_table(self.db_liquidity_table)
        create_benchmark_table(self.db_benchmark_table)

    def check_avg_slippage(self):
        start_date = str(
            datetime.date.fromisoformat(self.date)
            - datetime.timedelta(days=self.liquidity_consistency)
        )
        avg_liq_df = pd.read_sql(
            f"Select * from {self.db_liquidity_table} where date >= ? ",
            create_connection(db),
            index_col="date",
            parse_dates=["date"],
            params=[start_date],
        )
        avg_liq_df = avg_liq_df.ewm(span=3).mean().iloc[-1]
        avg_liq_df.index = convert_from_sql_strings(list(avg_liq_df.index))
        unacceptable_liq = avg_liq_df[avg_liq_df < self.max_slippage]
        for asset in unacceptable_liq.index:
            print(f"{asset} did not meet the average three month liquidity threshold")
        avg_liq_df = avg_liq_df[avg_liq_df >= self.max_slippage]
        return avg_liq_df

    def check_registered_assets(self):
        query = f"""{{indexAssets(where:{{index:\"{self.index_address}\"}}){{
        asset
        chainID
        }}}}"""
        resp = requests.post(
            self.subgraph_url,
            json={"query": query},
        )
        resp = resp.json()["data"]["indexAssets"]
        asset_dict = {}

        for asset_info in resp:
            asset = asset_info["asset"]
            chain_id = asset_info["chainID"]
            check = asset_dict.setdefault(chain_id, [])
            asset_dict[chain_id].append(asset)

        for id, data in self.results.iterrows():
            chain_id = self.chain_to_chain_id(data["blockchain"])
            if data["address"] not in asset_dict[chain_id]:
                print(
                    f"{id} needs to be registered on {data['blockchain']} with chain ID {chain_id} and address {data['address']}"
                )
