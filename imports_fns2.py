import pandas as pd
from datetime import datetime as dt
from connect_db import engine, read_sql_with_retry
import streamlit as st
import plotly.graph_objects as go
import calendar,ast
from datetime import datetime
import numpy as np

# Plotly Configurations
PLOTLY_GRAPH_CONFIG = {
    'displayModeBar': True,
    'responsive': True,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoscale',
        'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines',
        'sendDataToCloud', 'toggleHover', 'resetViewMap', 'toggleHover', 'toImage'
    ],
    'displaylogo': False
}
PLOTLY_TEMPLATE_LIGHT = 'plotly_white'

# Database Info
schemaName = 'Reference'
table_Name = 'FuturesExpire'

# Futures contract dictionary
futuresContractDict = {
    'F': {'abr': 'Jan', 'num': 1}, 'G': {'abr': 'Feb', 'num': 2}, 'H': {'abr': 'Mar', 'num': 3}, 'J': {'abr': 'Apr', 'num': 4},
    'K': {'abr': 'May', 'num': 5}, 'M': {'abr': 'Jun', 'num': 6}, 'N': {'abr': 'Jul', 'num': 7}, 'Q': {'abr': 'Aug', 'num': 8},
    'U': {'abr': 'Sep', 'num': 9}, 'V': {'abr': 'Oct', 'num': 10}, 'X': {'abr': 'Nov', 'num': 11}, 'Z': {'abr': 'Dec', 'num': 12}
}


# Function to determine the new 'No_Spread_Calc_Substring' variable
def assign_spread_type(type_str):
    if not isinstance(type_str, str):
        return "No_Spread_Calc_Substring"
    
    type_lower = type_str.lower()
    if "margin" in type_lower:
        return "Margin"
    elif "cross product" in type_lower:
        return "Cross Product"
    elif "flat" in type_lower:
        return "Flat"
    elif "crack" in type_lower:
        return "Crack"
    elif "box" in type_lower:
        return "Box"
    elif "calendar" in type_lower:
        return "Calendar"
    elif "quarter" in type_lower:
        return "Quarter"
    elif "arb" in type_lower:
        return "Arb"
    elif "lpg" in type_lower:
        return "LPG"
    elif "freight" in type_lower:
        return "Freight"
    else:
        return "No_Spread_Calc_Substring"

def generateYearList(contractMonthsList, yearOffsetList, current_calendar_month_num, selected_month, futuresContractDict):
    if len(contractMonthsList) != len(yearOffsetList):
        raise ValueError("contractMonthsList and yearOffsetList must be the same length.")

    current_year = dt.today().year
    
    # Determine the numerical value of the selected month
    selected_month_num = None
    if selected_month in futuresContractDict:
        selected_month_num = futuresContractDict[selected_month]['num']
    else:
        st.write("Error with selected month")
        pass # Handle this according to your error strategy

    # Calculate the base year for the contracts
    # If the current month is past the selected (roll) month, we should look for the next year's contract
    adjustment_year = 0
    if selected_month_num is not None and current_calendar_month_num >= selected_month_num:
        adjustment_year = 1
    

    year_list = []
    for offset in yearOffsetList:
        # Calculate the year for each ticker, applying the determined adjustment
        calculated_year = current_year + offset + adjustment_year
        year_list.append(str(calculated_year % 100).zfill(2))

    return year_list

def contractMonths(expireIn, contractRollIn, ContractMonthIn):
    tempExpire = expireIn[expireIn['Ticker'] == contractRollIn].copy()
    
    if not pd.api.types.is_datetime64_any_dtype(tempExpire['LastTrade']):
        tempExpire['LastTrade'] = pd.to_datetime(tempExpire['LastTrade'], format='%m/%d/%y', errors='coerce')

    tempExpire.set_index('LastTrade', inplace=True)
    tempExpire.dropna(subset=['LastTrade'], inplace=True)

    filtered_contracts = tempExpire[tempExpire.index > dt.today()].copy()

    expireDate = filtered_contracts[filtered_contracts['MonthCode'] == ContractMonthIn]

    if expireDate.empty:
        return None 
    return expireDate.iloc[0, :]

def generate_contract_data_sql_DEV(ticker, contractMonthsList, yearList, weights, conv, yearsBack):
    contract_data = {}

    past_date = dt.now().replace(year=dt.now().year - (yearsBack + 2))
    start_date_str = past_date.strftime('%Y-%m-%d')

    expireList = []

    for i, t in enumerate(ticker):
        contractMonth = contractMonthsList[i]
        
        current_contract_year_suffix = yearList[i]
        current_contract_full_year = 2000 + int(current_contract_year_suffix) if int(current_contract_year_suffix) < 50 else 1900 + int(current_contract_year_suffix)

        contractList = []
        for y_offset in range(yearsBack + 1):
            historical_full_year = current_contract_full_year - y_offset
            historical_year_suffix = str(historical_full_year % 100).zfill(2)
            contractList.append(f"{t}{contractMonth}{historical_year_suffix}")

        contract_list_str = ", ".join([f"'{c}'" for c in contractList])
        
        # Define placeholders here
        placeholders = ",".join([f"'{s}'" for s in contractList]) 
        
        sql_query = f"""
        SELECT trade_date, symbol, daily_close
        FROM MV_Prices.TimeSeries_DEV
        WHERE symbol IN ({placeholders})
        AND trade_date >= '{start_date_str}'
        ORDER BY symbol, trade_date
        """

        df = read_sql_with_retry(sql_query) 

        df.rename(columns={
            'trade_date': 'Date',
            'daily_close': 'close'
        }, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'])

        df['WeightedPrice'] = df['close'] * conv[i] * weights[i]

        contract_data[t] = {
            'Prices df': df,
            "ContractList": contractList,
            "Weights": weights[i],
            "Conversion": conv[i]
        }
        
        if i == 0:
            expireList = [c[-3:] for c in contractList]

    return contract_data, expireList


def generate_contract_data_sql_PROD(ticker, contractMonthsList, yearList, weights, conv, yearsBack):
    contract_data = {}
    
    # Calculate a common start date for fetching all relevant historical data to avoid multiple large queries
    past_date = dt.now().replace(year=dt.now().year - (yearsBack + 2))
    start_date_str = past_date.strftime('%Y-%m-%d')

    # Collect all unique contract symbols across all legs and all historical years
    all_contract_symbols_to_fetch = set()
    for i, t in enumerate(ticker):
        current_contract_year_suffix = yearList[i]
        current_contract_full_year = 2000 + int(current_contract_year_suffix) if int(current_contract_year_suffix) < 50 else 1900 + int(current_contract_year_suffix)

        for y_offset in range(yearsBack + 1):
            historical_full_year = current_contract_full_year - y_offset
            historical_year_suffix = str(historical_full_year % 100).zfill(2)
            all_contract_symbols_to_fetch.add(f"{t}{contractMonthsList[i]}{historical_year_suffix}")

    placeholders = ", ".join([f"'{c}'" for c in all_contract_symbols_to_fetch])
    
    # Fetch all data in one go
    sql_query_all = f"""
    SELECT trade_date, symbol, daily_close
    FROM MV_Prices.TimeSeries_PROD
    WHERE symbol IN ({placeholders})
    AND trade_date >= '{start_date_str}'
    ORDER BY symbol, trade_date
    """
    df_all_data = read_sql_with_retry(sql_query_all)
    df_all_data.rename(columns={'trade_date': 'Date', 'daily_close': 'close'}, inplace=True)
    df_all_data['Date'] = pd.to_datetime(df_all_data['Date'])

    print(f"\n--- Debug: Fetched ALL data (df_all_data) for all relevant contracts ---")
    print(df_all_data.head())
    print(df_all_data.info())

    # Now, process for each leg, applying specific weights and conversions
    for i, t in enumerate(ticker): # 't' is the ticker, e.g., '#ICENECM'
        contractMonth = contractMonthsList[i]
        current_contract_year_suffix = yearList[i]
        current_contract_full_year = 2000 + int(current_contract_year_suffix) if int(current_contract_year_suffix) < 50 else 1900 + int(current_contract_year_suffix)

        # Generate contract list for this specific leg for all historical years
        contractList_for_leg = []
        for y_offset in range(yearsBack + 1):
            historical_full_year = current_contract_full_year - y_offset
            historical_year_suffix = str(historical_full_year % 100).zfill(2)
            contractList_for_leg.append(f"{t}{contractMonth}{historical_year_suffix}")

        # Filter the master DataFrame for only the contracts relevant to the current leg
        df_for_this_leg = df_all_data[df_all_data['symbol'].isin(contractList_for_leg)].copy()

        # Apply the specific weight and conversion for THIS leg
        df_for_this_leg['WeightedPrice'] = df_for_this_leg['close'] * conv[i] * weights[i]

        print(f"\n--- Debug: DF for {t} (leg {i+1}) with its SPECIFIC WeightedPrice calculation ---")
        print(f"Using weight={weights[i]}, conversion={conv[i]} for contracts: {contractList_for_leg[:2]}...")
        print(df_for_this_leg[['Date', 'symbol', 'close', 'WeightedPrice']].head())
        print(df_for_this_leg['WeightedPrice'].describe())

        # Store data using a unique key for each leg (e.g., "Leg_0", "Leg_1")
        leg_key = f"Leg_{i}" # This ensures unique keys for each leg
        contract_data[leg_key] = {
            'Prices df': df_for_this_leg,
            "ContractList": contractList_for_leg, # Contracts relevant to THIS leg
            "Weights": weights[i],
            "Conversion": conv[i]
        }
    
    print("\n--- Debug: Final pricesDict structure (sample from generate_contract_data_sql_PROD) ---")
    for leg_key, data_value in contract_data.items():
        print(f"Leg Key: {leg_key}")
        print(data_value["Prices df"].head()) # Verify 'WeightedPrice' is present and correct for this leg
        print(f"Number of contracts for {leg_key}: {len(data_value['ContractList'])}")
        print("-" * 30)

    # Assuming expireList is only needed for the first leg's contract list for roll logic
    # More robust way to get expireList from the first leg's contractList after the loop
    expireList = []
    if 'Leg_0' in contract_data:
        expireList = [c[-3:] for c in contract_data['Leg_0']['ContractList']]


    return contract_data, expireList
def validate_contract_data(contract_data):
    contract_lengths = {ticker: len(data['ContractList']) for ticker, data in contract_data.items()}

    unique_lengths = set(contract_lengths.values())
    if len(unique_lengths) == 1:
        st.success(f"✅ All ContractList lengths are equal: {unique_lengths.pop()}")
    else:
        st.error("❌ ContractList lengths are not equal!")
        for ticker, length in contract_lengths.items():
            st.write(f"{ticker}: {length} contracts")
        st.stop()

    missing_data = []
    for ticker, data in contract_data.items():
        if 'Weights' not in data or 'Conversion' not in data:
            missing_data.append(ticker)

    if missing_data:
        st.error("❌ The following tickers are missing Weights or Conversion:")
        for ticker in missing_data:
            st.write(f"{ticker}: Missing {['Weights' if 'Weights' not in contract_data[ticker] else 'Conversion'][0]}")
        st.stop()
    else:
        st.success("✅ All tickers have Weights and Conversion.")

def get_required_symbols(spread_df, current_year_suffix):
    instruments = []

    for _, row in spread_df.iterrows():
        tickers = ast.literal_eval(row['tickerList'])
        months = ast.literal_eval(row['contractMonthsList'])
        # year_offsets should already be a list of integers here
        year_offsets = row['yearOffsetList'] 
        years_back = int(row['yearsBack'])

        for i, (ticker, month) in enumerate(zip(tickers, months)):
            base_full_year = (2000 + current_year_suffix) if current_year_suffix < 50 else (1900 + current_year_suffix)
            
            for y in range(years_back + 1):
                # year_offsets[i] is already an integer
                historical_full_year = base_full_year - y + year_offsets[i] 
                year_suffix = historical_full_year % 100
                symbol = f"{ticker}{month}{year_suffix:02d}"
                
                instruments.append({
                    'symbol': symbol,
                    'ticker': ticker,
                    'month': month,
                    'year_suffix': year_suffix,
                    'year_offset': year_offsets[i], 
                    'full_year': historical_full_year
                })

    return pd.DataFrame(instruments)

def load_filtered_prices_DEV(symbols):
    if not symbols:
        return pd.DataFrame()

    placeholders = ",".join([f"'{s}'" for s in symbols])
    query = f"""
        SELECT trade_date, symbol, daily_close
        FROM MV_Prices.TimeSeries_DEV
        WHERE symbol IN ({placeholders})
        ORDER BY trade_date, symbol
    """
    df = read_sql_with_retry(query)
    df.rename(columns={'trade_date': 'Date', 'daily_close': 'Close'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_filtered_prices_PROD(symbols):
    if not symbols:
        return pd.DataFrame()

    placeholders = ",".join([f"'{s}'" for s in symbols])
    query = f"""
        SELECT trade_date, symbol, daily_close
        FROM MV_Prices.TimeSeries_PROD
        WHERE symbol IN ({placeholders})
        ORDER BY trade_date, symbol
    """
    df = read_sql_with_retry(query)
    df.rename(columns={'trade_date': 'Date', 'daily_close': 'Close'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def process_spread_data_DEV(config_data_processed, df_prices, expire_df, today, futuresContractDict, current_calendar_year, current_calendar_month_num, selected_month):
   # variables = config_data_processed.to_dict()
    if isinstance(config_data_processed, dict):
        variables = config_data_processed
    else:
        variables = config_data_processed.to_dict()


    yearList = generateYearList(
        variables['contractMonthsList'],
        variables['yearOffsetList'],
        current_calendar_month_num,
        selected_month,
        futuresContractDict
    )

    pricesDict, expireList = generate_contract_data_sql_DEV(
        variables['tickerList'],
        variables['contractMonthsList'],
        yearList,
        variables['weightsList'],
        variables['convList'],
        variables['yearsBack']
    )

    validate_contract_data(pricesDict)

    combined_expire_symbols = [variables['rollFlag'] + exp for exp in expireList]

    expireMatrix = expire_df[expire_df['TickerMonthYear'].isin(combined_expire_symbols)].copy()
    expireMatrix['LastTrade'] = pd.to_datetime(expireMatrix['LastTrade'], format='%m/%d/%y', errors='coerce')
    expireMatrix.dropna(subset=['LastTrade'], inplace=True)

    if not expireMatrix.empty:
        expireMatrix['frontTicker'] = variables['tickerList'][0] + expireMatrix['MonthCode'] + expireMatrix['LastTrade'].dt.strftime('%y')
    else:
        st.warning("No relevant expiry data found for the roll flag.")
        return {}, pd.Series(dtype=float), pd.Series(dtype=float)

    spread_dict = {}
    if not pricesDict:
        st.warning("No price data dictionary generated for spread calculation.")
        return {}, pd.Series(dtype=float), pd.Series(dtype=float)

    first_ticker_data = next(iter(pricesDict.values()))
    if "ContractList" not in first_ticker_data or not first_ticker_data["ContractList"]:
        st.error("ContractList not found or is empty in pricesDict. Cannot determine num_historical_contracts.")
        return {}, pd.Series(dtype=float), pd.Series(dtype=float)

    num_historical_contracts = len(first_ticker_data["ContractList"])

    for i in range(num_historical_contracts):
        combined_df_for_year = pd.DataFrame()

        current_iter_contract_symbol = None
        if i < len(pricesDict[variables['tickerList'][0]]["ContractList"]):
            current_iter_contract_symbol = pricesDict[variables['tickerList'][0]]["ContractList"][i]

        if current_iter_contract_symbol is None:
            st.warning(f"Could not determine contract symbol for iteration {i}. Skipping.")
            continue

        for ticker_base, data in pricesDict.items():
            if i < len(data["ContractList"]):
                contract_symbol = data["ContractList"][i]
                temp_df = data["Prices df"][data["Prices df"]['symbol'] == contract_symbol][["Date", "WeightedPrice"]].copy()
                temp_df.set_index("Date", inplace=True)
                temp_df.rename(columns={"WeightedPrice": contract_symbol}, inplace=True)

                if combined_df_for_year.empty:
                    combined_df_for_year = temp_df
                else:
                    combined_df_for_year = combined_df_for_year.join(temp_df, how="outer")

        combined_df_for_year.dropna(inplace=True)
        combined_df_for_year["spread"] = combined_df_for_year.sum(axis=1, skipna=True)
        #st.write(combined_df_for_year)

        if not combined_df_for_year.empty:
            year_suffix = current_iter_contract_symbol[-2:]
            spread_year_full = 2000 + int(year_suffix) if int(year_suffix) < 50 else 1900 + int(year_suffix)
            spread_dict[str(spread_year_full)] = combined_df_for_year.copy()
        else:
            st.warning(f"No combined data for spread calculation for year derived from contract iteration {i}.")

    year_to_last_trade = expireMatrix.set_index("Year")["LastTrade"].to_dict()
    rows_to_drop = 5

    # ✅ Move this block up to avoid UnboundLocalError
    current_year_suffix_for_display = (
        (current_calendar_year + 1) % 100
        if current_calendar_month_num >= futuresContractDict[selected_month]['num']
        else current_calendar_year % 100
    )
    current_year_full_for_display = (
        2000 + int(str(current_year_suffix_for_display).zfill(2))
        if int(str(current_year_suffix_for_display).zfill(2)) < 50
        else 1900 + int(str(current_year_suffix_for_display).zfill(2))
    )
    shift_year_back = futuresContractDict[selected_month]['num'] > current_calendar_month_num

    filtered_spread_dict = {}
    for idx, (year_str, df) in enumerate(spread_dict.items()):
        year_int = int(year_str)
        if year_int in year_to_last_trade:
            last_trade_date = year_to_last_trade[year_int]
            df_filtered = df[df.index <= last_trade_date].copy()

            if last_trade_date < today and len(df_filtered) > rows_to_drop:
                df_filtered = df_filtered.iloc[:-rows_to_drop].copy()

            df_filtered['LastTrade'] = last_trade_date

            if not df_filtered.empty:
                is_latest_year = year_int == current_year_full_for_display

                try:
                    contract_month_num = futuresContractDict[selected_month]['num']

                    if is_latest_year:
                        # ✅ Fix: Extend filter window for latest year if month hasn't occurred yet
                        start_year = current_calendar_year - 1 if shift_year_back else current_calendar_year
                        filter_start_date = pd.to_datetime(f"{contract_month_num:02d}-01-{start_year}", format="%m-%d-%Y")

                        # st.info(
                        #     f"📅 Applying latest year filter: Start from {filter_start_date.date()} "
                        #     f"for year {year_int} (selected month: {selected_month}, shift back: {shift_year_back})"
                        # )

                        df_filtered = df_filtered[df_filtered.index >= filter_start_date].copy()

                    else:
                        adjusted_year = current_calendar_year - (idx - 1)
                        if shift_year_back:
                            adjusted_year -= 1

                        filter_start_date = pd.to_datetime(f"{contract_month_num:02d}-01-{adjusted_year - 1}", format="%m-%d-%Y")
                        filter_end_date = pd.to_datetime(f"{contract_month_num:02d}-01-{adjusted_year}", format="%m-%d-%Y")

                        # st.info(
                        #     f"📅 Applying historical filter: From {filter_start_date.date()} to {filter_end_date.date()} "
                        #     f"for year {year_int} (adjusted calendar year: {adjusted_year}, shift back: {shift_year_back})"
                        # )

                        df_filtered = df_filtered[
                            (df_filtered.index >= filter_start_date) & (df_filtered.index < filter_end_date)
                        ].copy()

                except Exception as e:
                    st.error(f"❌ Invalid date filter for year {year_int}: {e}")

                if not df_filtered.empty:
                    filtered_spread_dict[year_str] = df_filtered
        else:
            print(f"No LastTrade date found for year {year_str} in expire data. Skipping.")

    combined_spread_list = []
    current_year_spread_series = pd.Series(dtype=float)
    all_historical_spread_values = []

    for year_str, df_spread in filtered_spread_dict.items():
        if not df_spread.empty and 'spread' in df_spread.columns:
            df_copy = df_spread[['spread', 'LastTrade']].copy()
            df_copy["Year"] = year_str
            df_copy["Date"] = df_copy.index
            df_copy['TradingDay'] = (df_copy['Date'] - df_copy['Date'].min()).dt.days + 1

            if int(year_str) == current_year_full_for_display:
                current_year_spread_series = df_copy.set_index('Date')['spread']
                combined_spread_list.append(df_copy.reset_index(drop=True))
            else:
                combined_spread_list.append(df_copy.reset_index(drop=True))
                all_historical_spread_values.extend(df_copy['spread'].values)

    if combined_spread_list:
        df_out = pd.concat(combined_spread_list, ignore_index=True)
        df_out['InstrumentName'] = variables['Name']
        df_out['Group'] = variables['group']
        df_out['Region'] = variables['region']
        df_out['Month'] = variables['months']
        df_out['RollFlag'] = variables['rollFlag']
        df_out['Desc'] = variables['desc']
    else:
        df_out = pd.DataFrame()

    seasonal_chart_data = {}
    if not df_out.empty:
        for year, group_df in df_out.groupby('Year'):
            # Modified: Ensure 'Year' column is included in the seasonal_chart_data
            seasonal_chart_data[year] = group_df[['Date', 'spread', 'TradingDay', 'Year']].copy() 

        # Add 'Current' label for the latest year data if it exists, for histogram
        if str(current_year_full_for_display) in seasonal_chart_data:
            seasonal_chart_data['Current'] = seasonal_chart_data[str(current_year_full_for_display)].copy()


    all_spread_series_for_histogram = pd.Series(all_historical_spread_values, dtype=float).dropna()
    final_current_spread_series = current_year_spread_series.dropna()

    return seasonal_chart_data, all_spread_series_for_histogram, final_current_spread_series


def process_spread_data_PROD(config_data_processed, df_prices, expire_df, today, futuresContractDict, current_calendar_year, current_calendar_month_num, selected_month):
    if isinstance(config_data_processed, dict):
        variables = config_data_processed
    else:
        variables = config_data_processed.to_dict()


    yearList = generateYearList(
        variables['contractMonthsList'],
        variables['yearOffsetList'],
        current_calendar_month_num,
        selected_month,
        futuresContractDict
    )

    pricesDict, expireList = generate_contract_data_sql_PROD(
        variables['tickerList'],
        variables['contractMonthsList'],
        yearList,
        variables['weightsList'],
        variables['convList'],
        variables['yearsBack']
    )

    # validate_contract_data(pricesDict) # Assuming this function handles the new pricesDict structure

    combined_expire_symbols = [variables['rollFlag'] + exp for exp in expireList]

    expireMatrix = expire_df[expire_df['TickerMonthYear'].isin(combined_expire_symbols)].copy()
    expireMatrix['LastTrade'] = pd.to_datetime(expireMatrix['LastTrade'], format='%m/%d/%y', errors='coerce')
    expireMatrix.dropna(subset=['LastTrade'], inplace=True)

    if not expireMatrix.empty:
        # Assuming frontTicker is derived from the first leg's ticker and month code.
        # This might need adjustment if rollFlag is complex, but for now, it's fine for simple spreads.
        expireMatrix['frontTicker'] = variables['tickerList'][0] + expireMatrix['MonthCode'] + expireMatrix['LastTrade'].dt.strftime('%y')
    else:
        st.warning("No relevant expiry data found for the roll flag.")
        return {}, pd.Series(dtype=float), pd.Series(dtype=float)

    spread_dict = {}
    if not pricesDict:
        st.warning("No price data dictionary generated for spread calculation.")
        return {}, pd.Series(dtype=float), pd.Series(dtype=float)

    # Determine num_historical_contracts from the first leg (e.g., 'Leg_0')
    if 'Leg_0' not in pricesDict or "ContractList" not in pricesDict['Leg_0'] or not pricesDict['Leg_0']["ContractList"]:
        st.error("ContractList not found or is empty for Leg_0 in pricesDict. Cannot determine num_historical_contracts.")
        return {}, pd.Series(dtype=float), pd.Series(dtype=float)

    num_historical_contracts = len(pricesDict['Leg_0']["ContractList"])

    # print("\n--- Debug: pricesDict received by process_spread_data_PROD (sample) ---")
    # for leg_key, data_value in pricesDict.items():
    #     print(f"Leg Key: {leg_key}")
    #     print(data_value["Prices df"].head()) # CRUCIAL: Check 'WeightedPrice' column here
    #     print(f"Sample of WeightedPrice for {leg_key}: {data_value['Prices df']['WeightedPrice'].head().tolist()}")
    #     print("-" * 30)


    for i in range(num_historical_contracts):
        combined_df_for_year = pd.DataFrame()

        # Get the primary contract symbol for this historical year from Leg_0
        current_iter_contract_symbol = pricesDict['Leg_0']["ContractList"][i]

        print(f"\n--- Debug: Processing historical contract pair iteration {i} ---")
        print(f"Current iteration's primary contract symbol (from Leg_0): {current_iter_contract_symbol}")


        # Loop through each configured leg (e.g., 'Leg_0', 'Leg_1')
        for leg_key, data in pricesDict.items():
            if i < len(data["ContractList"]):
                # Get the specific historical contract symbol for this leg and this year iteration
                contract_symbol_for_this_leg_year = data["ContractList"][i]
                
                # Filter the master DataFrame for this leg to get only the data for this specific contract symbol
                temp_df_for_debug = data["Prices df"][data["Prices df"]['symbol'] == contract_symbol_for_this_leg_year].copy()
                
                print(f"--- Debug: Data for individual leg {leg_key}, contract {contract_symbol_for_this_leg_year} (before set_index/rename) ---")
                print(temp_df_for_debug[['Date', 'WeightedPrice']].head())
                
                temp_df = temp_df_for_debug[["Date", "WeightedPrice"]].copy()
                temp_df.set_index("Date", inplace=True)
                # Rename the column to the unique leg_key for correct joining
                temp_df.rename(columns={"WeightedPrice": leg_key}, inplace=True)

                if combined_df_for_year.empty:
                    combined_df_for_year = temp_df
                else:
                    # Join the current leg's data to the combined_df_for_year
                    combined_df_for_year = combined_df_for_year.join(temp_df, how="outer")
                    print(f"--- Debug: combined_df_for_year after joining {leg_key} ---")
                    print(combined_df_for_year.head())
                    print(combined_df_for_year.info()) # Check for NaNs after join
                    print(f"Columns: {combined_df_for_year.columns.tolist()}")


        print(f"--- Debug: combined_df_for_year before dropna() for iteration {i} ---")
        print(combined_df_for_year.head())
        print(combined_df_for_year.isnull().sum()) # Count NaNs per column

        combined_df_for_year.dropna(inplace=True)
        print(f"--- Debug: combined_df_for_year AFTER dropna() for iteration {i} ---")
        print(combined_df_for_year.head())
        print(combined_df_for_year.isnull().sum()) # Should be 0 NaNs if successful

        # Now, combined_df_for_year should have columns like 'Leg_0', 'Leg_1' etc., which can be summed
        combined_df_for_year["spread"] = combined_df_for_year.sum(axis=1, skipna=True)
        print(f"--- Debug: combined_df_for_year.head() AFTER spread calculation for iteration {i} ---")
        print(combined_df_for_year.head())
        print(combined_df_for_year['spread'].describe()) # Summary statistics for the calculated spread
        print(f"Sample of calculated spread values: {combined_df_for_year['spread'].head().tolist()}")
        # Check for infinite values or extremely large/small values
        print(f"Infinite values in spread: {np.isinf(combined_df_for_year['spread']).sum()}")
        print(f"NaN values in spread: {combined_df_for_year['spread'].isnull().sum()}")

        if not combined_df_for_year.empty:
            # Determine the year suffix from the primary contract symbol (Leg_0) for this iteration
            year_suffix = current_iter_contract_symbol[-2:]
            spread_year_full = 2000 + int(year_suffix) if int(year_suffix) < 50 else 1900 + int(year_suffix)
            spread_dict[str(spread_year_full)] = combined_df_for_year.copy()
        else:
            st.warning(f"No combined data for spread calculation for year derived from contract iteration {i}.")

    year_to_last_trade = expireMatrix.set_index("Year")["LastTrade"].to_dict()
    rows_to_drop = 5

    current_year_suffix_for_display = (
        (current_calendar_year + 1) % 100
        if current_calendar_month_num >= futuresContractDict[selected_month]['num']
        else current_calendar_year % 100
    )
    current_year_full_for_display = (
        2000 + int(str(current_year_suffix_for_display).zfill(2))
        if int(str(current_year_suffix_for_display).zfill(2)) < 50
        else 1900 + int(str(current_year_suffix_for_display).zfill(2))
    )
    shift_year_back = futuresContractDict[selected_month]['num'] > current_calendar_month_num

    filtered_spread_dict = {}
    for idx, (year_str, df) in enumerate(spread_dict.items()):
        year_int = int(year_str)
        if year_int in year_to_last_trade:
            last_trade_date = year_to_last_trade[year_int]
            df_filtered = df[df.index <= last_trade_date].copy()

            if last_trade_date < today and len(df_filtered) > rows_to_drop:
                df_filtered = df_filtered.iloc[:-rows_to_drop].copy()

            df_filtered['LastTrade'] = last_trade_date

            if not df_filtered.empty:
                is_latest_year = year_int == current_year_full_for_display

                try:
                    contract_month_num = futuresContractDict[selected_month]['num']

                    if is_latest_year:
                        start_year = current_calendar_year - 1 if shift_year_back else current_calendar_year
                        filter_start_date = pd.to_datetime(f"{contract_month_num:02d}-01-{start_year}", format="%m-%d-%Y")
                        df_filtered = df_filtered[df_filtered.index >= filter_start_date].copy()

                    else:
                        adjusted_year = current_calendar_year - (idx - 1)
                        if shift_year_back:
                            adjusted_year -= 1

                        filter_start_date = pd.to_datetime(f"{contract_month_num:02d}-01-{adjusted_year - 1}", format="%m-%d-%Y")
                        filter_end_date = pd.to_datetime(f"{contract_month_num:02d}-01-{adjusted_year}", format="%m-%d-%Y")

                        df_filtered = df_filtered[
                            (df_filtered.index >= filter_start_date) & (df_filtered.index < filter_end_date)
                        ].copy()

                except Exception as e:
                    st.error(f"❌ Invalid date filter for year {year_int}: {e}")

                if not df_filtered.empty:
                    filtered_spread_dict[year_str] = df_filtered
        else:
            st.warning(f"⚠️ No LastTrade date found for year {year_str} in expire data. Skipping.")

    combined_spread_list = []
    current_year_spread_series = pd.Series(dtype=float)
    all_historical_spread_values = []

    for year_str, df_spread in filtered_spread_dict.items():
        if not df_spread.empty and 'spread' in df_spread.columns:
            df_copy = df_spread[['spread', 'LastTrade']].copy()
            df_copy["Year"] = year_str
            df_copy["Date"] = df_copy.index
            df_copy['TradingDay'] = (df_copy['Date'] - df_copy['Date'].min()).dt.days + 1

            if int(year_str) == current_year_full_for_display:
                current_year_spread_series = df_copy.set_index('Date')['spread']
                combined_spread_list.append(df_copy.reset_index(drop=True))
            else:
                combined_spread_list.append(df_copy.reset_index(drop=True))
                all_historical_spread_values.extend(df_copy['spread'].values)

    if combined_spread_list:
        df_out = pd.concat(combined_spread_list, ignore_index=True)
        df_out['InstrumentName'] = variables['Name']
        df_out['Group'] = variables['group']
        df_out['Region'] = variables['region']
        df_out['Month'] = variables['months']
        df_out['RollFlag'] = variables['rollFlag']
        df_out['Desc'] = variables['desc']
    else:
        df_out = pd.DataFrame()

    seasonal_chart_data = {}
    if not df_out.empty:
        for year, group_df in df_out.groupby('Year'):
            seasonal_chart_data[year] = group_df[['Date', 'spread', 'TradingDay', 'Year']].copy() 

        if str(current_year_full_for_display) in seasonal_chart_data:
            seasonal_chart_data['Current'] = seasonal_chart_data[str(current_year_full_for_display)].copy()


    all_spread_series_for_histogram = pd.Series(all_historical_spread_values, dtype=float).dropna()
    final_current_spread_series = current_year_spread_series.dropna()

    print("\n--- Debug: Final all_spread_series_for_histogram.head() ---")
    print(all_spread_series_for_histogram.head())
    print(all_spread_series_for_histogram.describe())
    print(f"Final all_spread_series_for_histogram size: {len(all_spread_series_for_histogram)}")

    print("\n--- Debug: Final final_current_spread_series.head() ---")
    print(final_current_spread_series.head())
    print(final_current_spread_series.describe())
    print(f"Final final_current_spread_series size: {len(final_current_spread_series)}")

    return seasonal_chart_data, all_spread_series_for_histogram, final_current_spread_series

def create_dash_style_seasonal_chart(seasonal_data, title="Plot", units=""):
    fig = go.Figure()

    year_numbers = []
    for label in seasonal_data.keys():
        if isinstance(label, str) and label.isdigit():
            year_numbers.append(int(label))
    current_year_label = str(max(year_numbers)) if year_numbers else None
    
    if "Current" in seasonal_data:
        current_year_label = "Current"


    for label, df_original in seasonal_data.items():
        df = df_original.copy()
        df = df.drop_duplicates(subset=['Date']).sort_values('Date')
        
        line_color = "black" if label == current_year_label else (
            "blue" if label == 'Median' else (
            "purple" if label == 'Mean' else None
        ))
        line_width = 3 if label == current_year_label else (2 if label in ['Median', 'Mean'] else 1.5)
        line_dash = 'dot' if label in ['Median', 'Mean'] else None
        opacity = 1.0 if label == current_year_label else (0.8 if label in ['Median', 'Mean'] else 0.6)

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["spread"],
            mode="lines",
            name=str(label),
            line=dict(
                color=line_color,
                width=line_width,
                dash=line_dash
            ),
            opacity=opacity,
            connectgaps=False
        ))

    fig.update_layout(
        title_text=f"<b>Seasonal Price Evolution ({units})</b>",
        title_x=0.5,
        hovermode="x unified",
        font_color='white',
        paper_bgcolor='white',
        template=PLOTLY_TEMPLATE_LIGHT,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(
        tickformat="%b %d",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey'
    )
    fig.update_yaxes(
        title_text=units or "Price",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGrey'
    )

    return fig

def create_dash_style_histogram(seasonal_data, spread_data, title): # Added seasonal_data parameter
    fig = go.Figure()

    if spread_data.empty:
        fig.add_annotation(
            text="No data available for histogram",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    fig.add_trace(go.Histogram(
        x=spread_data,
        marker_color='#1f77b4',
        nbinsx=100,
        name='Spread Distribution',
        hovertemplate="<b>Range:</b> %{x:.2f}<br><b>Frequency:</b> %{y}<extra></extra>",
        opacity=0.7
    ))

    median_val = spread_data.median()
    std_dev = spread_data.std()
    mean_val = spread_data.mean()
    min_val = spread_data.min()
    max_val = spread_data.max()
    
    # Get latest_value from 'Current' seasonal data if available
    latest_value = None
    # Check if 'Current' key exists and is not an empty DataFrame
    if 'Current' in seasonal_data and not seasonal_data['Current'].empty:
        latest_value = seasonal_data['Current']['spread'].iloc[-1]
    
    stats_text = (
        f"<b>Statistics:</b><br>"
        f"Median: {median_val:.2f}<br>"
        f"Mean: {mean_val:.2f}<br>"
        f"Std Dev: {std_dev:.2f}<br>"
        f"Min: {min_val:.2f}<br>"
        f"Max: {max_val:.2f}"
    )
    if latest_value is not None:
        stats_text += f"<br>Latest: {latest_value:.2f}"

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        text=stats_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=5,
        font=dict(size=12, color="black")
    )

    fig.add_vline(x=median_val, line=dict(dash="dash", color="green", width=2),
                  annotation_text=f"Median: {median_val:.2f}",
                  annotation_position="top left",
                  annotation_font_color="green")

    fig.add_vline(x=median_val + std_dev, line=dict(dash="dot", color="orange", width=1.5),
                  annotation_text=f"+1 SD: {(median_val + std_dev):.2f}",
                  annotation_position="top right",
                  annotation_font_color="orange")

    fig.add_vline(x=median_val - std_dev, line=dict(dash="dot", color="orange", width=1.5),
                  annotation_text=f"-1 SD: {(median_val - std_dev):.2f}",
                  annotation_position="bottom left",
                  annotation_font_color="orange")

    fig.add_vline(x=median_val + (2 * std_dev), line=dict(dash="dot", color="red", width=1.5),
                  annotation_text=f"+2 SD: {(median_val + (2 * std_dev)):.2f}",
                  annotation_position="top right",
                  annotation_font_color="red")

    fig.add_vline(x=median_val - (2 * std_dev), line=dict(dash="dot", color="red", width=1.5),
                  annotation_text=f"-2 SD: {(median_val - (2 * std_dev)):.2f}",
                  annotation_position="bottom left",
                  annotation_font_color="red")

    if latest_value is not None:
        fig.add_vline(x=latest_value, line=dict(dash="solid", color="blue", width=3),
                      annotation_text=f"Latest: {latest_value:.2f}",
                      annotation_position="bottom right",
                      annotation_font_color="blue")

    fig.update_layout(
        title="Distribution of Spread (Histogram)",
        xaxis_title="Spread",
        yaxis_title="Frequency",
        template=PLOTLY_TEMPLATE_LIGHT,
        showlegend=True,
        height=600
    )

    return fig


def create_dash_style_seasonal_chart2(seasonal_data, title="", units="", selected_month_code="F"):
    """
    Create a seasonal chart with months reordered starting from the selected contract month.
    
    Args:
        seasonal_data: Dict of {year_label: DataFrame} containing Date and spread columns
        title: Chart title
        units: Y-axis units label
        selected_month_code: Contract month code (F, G, H, etc.) to start the x-axis from
    """
    fig = go.Figure()

    # Constants for synthetic timeline
    BASE_YEAR = 2000
    DAYS_PER_MONTH = 30

    # Get month ordering starting from selected contract month
    start_month_num = futuresContractDict[selected_month_code]['num']
    reordered_months = _get_reordered_months(start_month_num)
    month_labels = [calendar.month_abbr[month] for month in reordered_months]

    # Create mapping for date transformation
    month_to_display_index = {month: idx for idx, month in enumerate(reordered_months)}

    # Identify current year for special styling
    current_year_label = _identify_current_year(seasonal_data)

    # Plot each year's data
    for year_label, year_df in seasonal_data.items():
        if year_label in ['Mean', 'Median', 'Current']:  # Skip special labels
            continue

        cleaned_df = _prepare_year_data(year_df)
        if cleaned_df.empty:
            continue

        # Transform dates to synthetic display timeline
        display_dates = _transform_dates_to_display_timeline(
            cleaned_df["Date"], month_to_display_index, BASE_YEAR, DAYS_PER_MONTH
        )

        # Apply styling based on year type
        style_config = _get_line_style(year_label, current_year_label)

        fig.add_trace(go.Scatter(
            x=display_dates,
            y=cleaned_df["spread"],
            mode="lines",
            name=str(year_label),
            line=style_config['line'],
            opacity=style_config['opacity'],
            connectgaps=False,
            hovertemplate='%{y:.4f} ' + units + '<extra></extra>'  # Show only y, hide x
        ))

    # Configure layout and axes
    _configure_chart_layout(fig, title, units, month_labels, BASE_YEAR, DAYS_PER_MONTH)

    return fig

def create_volatility_chart(seasonal_data, title="", selected_month_code="F"):
    """
    Create a volatility chart showing 20-day rolling volatility of spread changes.
    
    Args:
        seasonal_data: Dict of {year_label: DataFrame} containing Date and spread columns
        title: Chart title
        selected_month_code: Contract month code for x-axis ordering
    """
    # Constants
    BASE_YEAR = 2000
    DAYS_PER_MONTH = 30
    VOLATILITY_WINDOW = 20
    VOLATILITY_MULTIPLIER = 2
    
    # Get month ordering (same as seasonal chart)
    start_month_num = futuresContractDict[selected_month_code]['num']
    reordered_months = _get_reordered_months(start_month_num)
    month_labels = [calendar.month_abbr[month] for month in reordered_months]
    month_to_display_index = {month: idx for idx, month in enumerate(reordered_months)}
    
    # Build combined spread DataFrame, excluding special labels like 'Mean', 'Median', 'Current'
    spread_df = _build_spread_dataframe(
        {k: v for k, v in seasonal_data.items() if k not in ['Mean', 'Median', 'Current']}
    )
    
    # Initialize latest_volatility_value as NaN
    latest_volatility_value = np.nan

    if spread_df.empty:
        # Create an empty figure with a warning if no data is available
        fig = go.Figure()
        fig.add_annotation(
            text="No valid spread data found for volatility calculation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig, latest_volatility_value # Return figure and NaN volatility
    
    # Calculate rolling volatility
    volatility_df = _calculate_rolling_volatility(spread_df, VOLATILITY_WINDOW, VOLATILITY_MULTIPLIER)
    
    # Identify current year for styling
    current_year_label = _identify_current_year(seasonal_data)
    
    # Get the latest volatility value for the current year, if available
    if current_year_label in volatility_df.columns:
        current_year_volatility_series = volatility_df[current_year_label].dropna()
        if not current_year_volatility_series.empty:
            latest_volatility_value = current_year_volatility_series.iloc[-1]
    
    fig = go.Figure()
    
    # Plot volatility for each year
    for year_label in volatility_df.columns:
        year_volatility = volatility_df[year_label].dropna()
        if year_volatility.empty:
            continue
        
        # Transform dates to display timeline
        display_dates = _transform_dates_to_display_timeline(
            year_volatility.index, month_to_display_index, BASE_YEAR, DAYS_PER_MONTH
        )
        
        # Apply same styling as seasonal chart
        style_config = _get_line_style(year_label, current_year_label)
        
        fig.add_trace(go.Scatter(
            x=display_dates,
            y=year_volatility.values,
            mode="lines",
            name=str(year_label),
            line=style_config['line'],
            opacity=style_config['opacity'],
            connectgaps=False
        ))
    
    # Configure layout (similar to seasonal chart but with volatility-specific y-axis)
    _configure_chart_layout(fig, title, "Spread Volatility", month_labels, BASE_YEAR, DAYS_PER_MONTH)
    
    return fig, latest_volatility_value

# Helper functions for cleaner code organization
def filter_seasonal_data_for_latest_year(seasonal_data, current_date=None):
    """
    Filter seasonal data to keep only observations from the latest/current year.
    
    Args:
        seasonal_data: Dict of {year_label: DataFrame} containing Date and spread columns
        current_date: Reference date to determine what constitutes "latest" (defaults to today)
    
    Returns:
        Dict with only the latest year's data, plus any statistical measures (Mean, Median)
    """
    if current_date is None:
        current_date = pd.Timestamp.now()
    
    # Separate statistical measures from year data
    statistical_keys = {'Mean', 'Median', 'Average', 'Std', 'StdDev'}
    year_data = {}
    stats_data = {}
    
    for key, df in seasonal_data.items():
        if key in statistical_keys:
            stats_data[key] = df
        else:
            year_data[key] = df
    
    # Find the latest year
    if "Current" in year_data:
        latest_key = "Current"
        latest_data = {latest_key: year_data[latest_key]}
    else:
        # Find the most recent year based on the maximum date in each year's data
        latest_key = None
        latest_max_date = pd.Timestamp.min
        
        for year_label, year_df in year_data.items():
            if 'Date' in year_df.columns and not year_df.empty:
                max_date_in_year = year_df['Date'].max()
                if max_date_in_year > latest_max_date:
                    latest_max_date = max_date_in_year
                    latest_key = year_label
        
        latest_data = {latest_key: year_data[latest_key]} if latest_key else {}
    
    # Combine latest year data with statistical measures
    return {**latest_data, **stats_data}


def filter_seasonal_data_by_date_range(seasonal_data, start_date, end_date):
    """
    Filter seasonal data to keep only observations within a specific date range.
    
    Args:
        seasonal_data: Dict of {year_label: DataFrame} containing Date and spread columns
        start_date: Start date for filtering
        end_date: End date for filtering
    
    Returns:
        Dict with filtered data for each year
    """
    filtered_data = {}
    
    for year_label, year_df in seasonal_data.items():
        if 'Date' not in year_df.columns:
            continue
            
        # Filter by date range
        mask = (year_df['Date'] >= start_date) & (year_df['Date'] <= end_date)
        filtered_df = year_df[mask].copy()
        
        # Only include if there's data after filtering
        if not filtered_df.empty:
            filtered_data[year_label] = filtered_df
    
    return filtered_data


def filter_seasonal_data_by_month_range(seasonal_data, start_month, end_month):
    """
    Filter seasonal data to keep only observations within a specific month range.
    Useful for seasonal analysis (e.g., only show data from March to September).
    
    Args:
        seasonal_data: Dict of {year_label: DataFrame} containing Date and spread columns
        start_month: Start month (1-12)
        end_month: End month (1-12)
    
    Returns:
        Dict with filtered data for each year
    """
    filtered_data = {}
    
    for year_label, year_df in seasonal_data.items():
        if 'Date' not in year_df.columns:
            continue
            
        # Extract month from dates
        months = year_df['Date'].dt.month
        
        # Handle cross-year month ranges (e.g., Nov to Mar)
        if start_month <= end_month:
            mask = (months >= start_month) & (months <= end_month)
        else:
            mask = (months >= start_month) | (months <= end_month)
        
        filtered_df = year_df[mask].copy()
        
        # Only include if there's data after filtering
        if not filtered_df.empty:
            filtered_data[year_label] = filtered_df
    
    return filtered_data


def _get_reordered_months(start_month_num):
    """Reorder months starting from the given month number."""
    return list(range(start_month_num, 13)) + list(range(1, start_month_num))


def _identify_current_year(seasonal_data):
    """
    Identify the current year label for special styling.
    Prioritizes 'Current' key if it's a valid DataFrame with a 'Year' column.
    Falls back to the numerically largest year if 'Current' is not ideal.
    """
    # Check for 'Current' key first
    if "Current" in seasonal_data:
        current_df = seasonal_data["Current"]
        if isinstance(current_df, pd.DataFrame) and not current_df.empty and 'Year' in current_df.columns:
            return str(current_df['Year'].iloc[0])

    # Fallback to finding the maximum year if 'Current' is not reliable
    year_numbers = []
    for label, data_item in seasonal_data.items():
        if isinstance(label, str) and str(label).isdigit() and isinstance(data_item, pd.DataFrame) and not data_item.empty:
            # Only consider numeric year labels that correspond to non-empty DataFrames
            year_numbers.append(int(label))
    
    return str(max(year_numbers)) if year_numbers else None # Return as string, or None if no valid years


def _prepare_year_data(year_df):
    """Clean and prepare year data for plotting."""
    if 'spread' not in year_df.columns or 'Date' not in year_df.columns:
        return pd.DataFrame()
    
    return (year_df[['Date', 'spread']]
            .drop_duplicates(subset=['Date'])
            .sort_values('Date')
            .dropna(subset=['spread']))


def _transform_dates_to_display_timeline(dates, month_to_display_index, base_year, days_per_month):
    """Transform actual dates to synthetic numeric timeline (e.g., day offsets)."""
    display_dates = []
    for date in dates:
        # Get the synthetic month index from reordered calendar
        month_idx = month_to_display_index.get(date.month, date.month - 1)  # fallback
        days_offset = month_idx * days_per_month + date.day - 1
        display_dates.append(days_offset)  # Use int instead of Timestamp
    return display_dates

def _get_line_style(year_label, current_year_label):
    """Get line styling and opacity based on year type."""
    # Current year gets prominent black line
    if str(year_label) == str(current_year_label): # Ensure string comparison
        return {
            'line': {'color': 'white', 'width': 3},
            'opacity': 1.0
        }
    
    # Statistical lines get special colors and dash patterns
    if year_label == 'Median':
        return {
            'line': {'color': 'blue', 'width': 2, 'dash': 'dot'},
            'opacity': 0.8
        }
    
    if year_label == 'Mean':
        return {
            'line': {'color': 'purple', 'width': 2, 'dash': 'dot'},
            'opacity': 0.8
        }
    
    # Historical years get default styling
    return {
        'line': {'width': 1.5},
        'opacity': 0.6
    }


def _build_spread_dataframe(seasonal_data):
    """Build a combined DataFrame with spread data from all years."""
    spread_series = {}
    
    for year_label, year_df in seasonal_data.items():
        if 'spread' not in year_df.columns or 'Date' not in year_df.columns:
            continue
            
        # Prepare clean data indexed by date
        clean_data = (year_df[['Date', 'spread']]
                     .drop_duplicates(subset=['Date'])
                     .sort_values('Date')
                     .set_index('Date')
                     .dropna())
        
        if not clean_data.empty:
            spread_series[year_label] = clean_data['spread']
    
    return pd.concat(spread_series, axis=1) if spread_series else pd.DataFrame()

def _calculate_rolling_volatility(df, window, multiplier):
    padded_df = {}

    for year in df.columns:
        series = df[year]
        if series.empty:
            continue
        
        # Pad end of year with start-of-year data
        padding = series.head(window - 1)
        extended_series = pd.concat([series, padding])
        
        # Calculate volatility
        diff = extended_series.diff()
        vol = diff.rolling(window).std() * multiplier
        
        # Trim to original length
        padded_df[year] = vol.iloc[:len(series)]

    return pd.DataFrame(padded_df)



def _configure_chart_layout(fig, title, y_axis_label, month_labels, base_year, days_per_month):
    """Configure the chart layout, axes, and styling (Dark Theme)."""
    fig.update_layout(
        title_text=f"<b>{title}</b>",
        title_x=0.5,
        hovermode="x unified",
        font_color='white',
        paper_bgcolor='#1e1e1e',   # Dark background
        plot_bgcolor='#1e1e1e',
        width=900,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        template=None,  # Remove default template to allow full dark styling
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="white"),
            bgcolor="rgba(0,0,0,0)"
        )
    )

    # X-axis: Use synthetic tick positions (middle of each synthetic month)
    tick_positions = [i * days_per_month + days_per_month // 2 for i in range(len(month_labels))]

    fig.update_xaxes(
        tickvals=tick_positions,
        ticktext=month_labels,
        showgrid=True,
        gridwidth=1,
        gridcolor='#444',         # Subtle grid
        linecolor='white',
        title_text="Month",
        tickfont=dict(color='white'),
        title_font=dict(color='white'),
        hoverformat=None,
        tickformat=None
    )

    fig.update_yaxes(
        title_text=y_axis_label,
        showgrid=True,
        gridwidth=1,
        gridcolor='#444',
        linecolor='white',
        tickfont=dict(color='white'),
        title_font=dict(color='white')
    )

def filter_data_by_synthetic_month_day(seasonal_data_original, current_year_label, filter_start_synthetic_date, filter_end_synthetic_date):
    filtered_seasonal_data = {}
    all_historical_spread_values_filtered = []
    filtered_current_spread_series_output = pd.Series(dtype=float)

    # Convert start and end filter dates to month-day tuples for numerical comparison
    start_filter_month_day = (filter_start_synthetic_date.month, filter_start_synthetic_date.day)
    end_filter_month_day = (filter_end_synthetic_date.month, filter_end_synthetic_date.day)

    for year_label, df_original in seasonal_data_original.items():
        # Skip special labels or non-DataFrame entries that are not actual year data
        if year_label in ['Mean', 'Median']:
            continue
        
        if not isinstance(df_original, pd.DataFrame) or df_original.empty:
            continue
        
        if 'Date' not in df_original.columns or 'spread' not in df_original.columns:
            continue

        df = df_original.copy()
        # Extract month and day as integers for robust comparison
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

        # Create month-day tuple for each row for comparison
        df['MonthDayTuple'] = list(zip(df['Month'], df['Day']))

        # Apply filtering logic using tuple comparison
        if start_filter_month_day <= end_filter_month_day:
            # Standard range (e.g., Jan to March)
            mask = (df['MonthDayTuple'] >= start_filter_month_day) & \
                   (df['MonthDayTuple'] <= end_filter_month_day)
        else:
            # Wrapped range (e.g., Nov to Feb)
            # Days are either from start_month_day to Dec 31, OR from Jan 1 to end_month_day
            mask = (df['MonthDayTuple'] >= start_filter_month_day) | \
                   (df['MonthDayTuple'] <= end_filter_month_day)
        
        filtered_df = df[mask].copy()

        if not filtered_df.empty:
            filtered_seasonal_data[year_label] = filtered_df
            # If this is the current year's data, populate the specific series for current year stats
            if str(year_label) == str(current_year_label):
                filtered_current_spread_series_output = filtered_df.set_index('Date')['spread']
            else:
                # Accumulate historical filtered values for the overall histogram
                all_historical_spread_values_filtered.extend(filtered_df['spread'].values)
        
    all_spread_series_filtered_output = pd.Series(all_historical_spread_values_filtered, dtype=float).dropna()

    return filtered_seasonal_data, all_spread_series_filtered_output, filtered_current_spread_series_output

def filter_data_by_synthetic_month_day(seasonal_data_original, current_year_label, filter_start_synthetic_date, filter_end_synthetic_date):
    filtered_seasonal_data = {}
    all_historical_spread_values_filtered = []
    filtered_current_spread_series_output = pd.Series(dtype=float)

    # Convert start and end filter dates to month-day tuples for numerical comparison
    start_filter_month_day = (filter_start_synthetic_date.month, filter_start_synthetic_date.day)
    end_filter_month_day = (filter_end_synthetic_date.month, filter_end_synthetic_date.day)

    for year_label, df_original in seasonal_data_original.items():
        # Skip special labels or non-DataFrame entries that are not actual year data
        if year_label in ['Mean', 'Median']:
            continue
        
        if not isinstance(df_original, pd.DataFrame) or df_original.empty:
            continue
        
        if 'Date' not in df_original.columns or 'spread' not in df_original.columns:
            continue

        df = df_original.copy()
        # Extract month and day as integers for robust comparison
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

        # Create month-day tuple for each row for comparison
        df['MonthDayTuple'] = list(zip(df['Month'], df['Day']))

        # Apply filtering logic using tuple comparison
        if start_filter_month_day <= end_filter_month_day:
            # Standard range (e.g., Jan to March)
            mask = (df['MonthDayTuple'] >= start_filter_month_day) & \
                   (df['MonthDayTuple'] <= end_filter_month_day)
        else:
            # Wrapped range (e.g., Nov to Feb)
            # Days are either from start_month_day to Dec 31, OR from Jan 1 to end_month_day
            mask = (df['MonthDayTuple'] >= start_filter_month_day) | \
                   (df['MonthDayTuple'] <= end_filter_month_day)
        
        filtered_df = df[mask].copy()

        if not filtered_df.empty:
            filtered_seasonal_data[year_label] = filtered_df
            # If this is the current year's data, populate the specific series for current year stats
            if str(year_label) == str(current_year_label):
                filtered_current_spread_series_output = filtered_df.set_index('Date')['spread']
            else:
                # Accumulate historical filtered values for the overall histogram
                all_historical_spread_values_filtered.extend(filtered_df['spread'].values)
        
    all_spread_series_filtered_output = pd.Series(all_historical_spread_values_filtered, dtype=float).dropna()

    return filtered_seasonal_data, all_spread_series_filtered_output, filtered_current_spread_series_output


def create_monthly_performance_heatmap(seasonal_data, title="Monthly Performance Heatmap", units="Performance"):
    """
    Creates a monthly performance heatmap from seasonal data.
    
    Parameters:
    seasonal_data (dict): Dictionary where keys are year labels and values are pandas Series/DataFrame
                         with datetime index containing performance values
    title (str): Title for the heatmap
    units (str): Units label for the colorbar
    
    Returns:
    plotly.graph_objects.Figure: The heatmap figure
    """
    print(f"Debug - Input seasonal_data type: {type(seasonal_data)}")
    print(f"Debug - Input seasonal_data keys: {list(seasonal_data.keys()) if seasonal_data else 'No keys'}")
    
    if not seasonal_data or len(seasonal_data) == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(text="No data available for heatmap", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=400, template='plotly_white')
        return fig
    
    # Prepare data for heatmap
    heatmap_data = []
    
    for year_label, data in seasonal_data.items():
        print(f"Debug - Processing year: {year_label}, data type: {type(data)}")
        
        if data is None:
            print(f"Debug - Data is None for year {year_label}")
            continue
            
        if hasattr(data, 'empty') and data.empty:
            print(f"Debug - Data is empty for year {year_label}")
            continue
        
        # Handle both Series and DataFrame
        if isinstance(data, pd.DataFrame):
            print(f"Debug - DataFrame columns: {data.columns.tolist()}")
            # Use the first numeric column or a specific column for performance calculation
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                print(f"Debug - No numeric columns found in DataFrame for year {year_label}")
                continue
            # Use the first numeric column (you may want to specify which column to use)
            series = data[numeric_cols[0]]
            print(f"Debug - Using column '{numeric_cols[0]}' for year {year_label}")
        else:
            series = data
        
        # Extract year from label
        try:
            year = str(year_label)
            print(f"Debug - Year string: {year}")
        except:
            print(f"Debug - Could not convert year_label to string: {year_label}")
            continue
        
        # Ensure index is datetime
        if not isinstance(series.index, pd.DatetimeIndex):
            print(f"Debug - Converting index to datetime for year {year}")
            try:
                series.index = pd.to_datetime(series.index)
            except Exception as e:
                print(f"Debug - Failed to convert index to datetime: {e}")
                continue
        
        # Convert series values to numeric
        try:
            series_numeric = pd.to_numeric(series, errors='coerce')
            print(f"Debug - Series numeric conversion successful, shape: {series_numeric.shape}")
        except Exception as e:
            print(f"Debug - Failed to convert series to numeric: {e}")
            continue
        
        # Drop NaN values after conversion
        series_numeric = series_numeric.dropna()
        print(f"Debug - After dropping NaN, shape: {series_numeric.shape}")
        
        if series_numeric.empty:
            print(f"Debug - No valid numeric data for year {year}")
            continue
        
        # Group by month and calculate monthly performance
        monthly_perf = []
        for month in range(1, 13):
            try:
                month_data = series_numeric[series_numeric.index.month == month]
                if not month_data.empty:
                    # Calculate monthly performance as change from first to last value
                    if len(month_data) > 1:
                        performance = month_data.iloc[-1] - month_data.iloc[0]
                    else:
                        performance = month_data.iloc[0]
                    monthly_perf.append(performance)
                    print(f"Debug - Month {month}: performance = {performance}")
                else:
                    monthly_perf.append(np.nan)
                    print(f"Debug - Month {month}: no data")
            except Exception as e:
                print(f"Debug - Error processing month {month}: {e}")
                monthly_perf.append(np.nan)
        
        for month_idx, perf in enumerate(monthly_perf):
            if not np.isnan(perf):
                heatmap_data.append({
                    'Year': year,
                    'Month': month_idx + 1,
                    'Month_Name': calendar.month_abbr[month_idx + 1],
                    'Performance': perf
                })
    
    print(f"Debug - Total heatmap data points: {len(heatmap_data)}")
    
    if not heatmap_data:
        fig = go.Figure()
        fig.add_annotation(text="No valid numeric data found for heatmap", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=400, template='plotly_white')
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(heatmap_data)
    print(f"Debug - DataFrame shape: {df.shape}")
    print(f"Debug - DataFrame columns: {df.columns.tolist()}")
    
    # Pivot for heatmap
    try:
        pivot_df = df.pivot(index='Year', columns='Month_Name', values='Performance')
        print(f"Debug - Pivot successful, shape: {pivot_df.shape}")
    except Exception as e:
        print(f"Debug - Pivot failed: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating pivot table: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=400, template='plotly_white')
        return fig
    
    # Reorder columns to calendar order
    month_order = [calendar.month_abbr[i] for i in range(1, 13)]
    pivot_df = pivot_df.reindex(columns=month_order)
    
    # Create heatmap
    try:
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<br><extra></extra>',
            colorbar=dict(title=units),
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Year",
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
            height=400,
            template='plotly_white'
        )
        
        print("Debug - Heatmap created successfully")
        return fig
        
    except Exception as e:
        print(f"Debug - Error creating heatmap: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating heatmap: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, height=400, template='plotly_white')
        return fig


def create_daily_summary_table(seasonal_data, selected_year=None, max_days=30):
    """
    Creates a daily summary statistics table from seasonal data.
    
    Parameters:
    seasonal_data (dict): Dictionary where keys are year labels and values are pandas Series/DataFrame
                         with datetime index containing performance values
    selected_year (str): Specific year to show, if None shows current/latest year
    max_days (int): Maximum number of days to show in table
    
    Returns:
    plotly.graph_objects.Figure: The table figure
    """
    print(f"Debug - Table input seasonal_data type: {type(seasonal_data)}")
    print(f"Debug - Selected year: {selected_year}")
    print(f"Debug - Max days: {max_days}")
    
    if not seasonal_data or len(seasonal_data) == 0:
        return _create_error_table("No data available")
    
    # Select year to display
    if selected_year and selected_year in seasonal_data:
        year_data = seasonal_data[selected_year]
        year_label = selected_year
    else:
        # Use the most recent year available
        try:
            year_label = max(seasonal_data.keys(), key=str)
            year_data = seasonal_data[year_label]
        except Exception as e:
            print(f"Debug - Error finding max year: {e}")
            return _create_error_table("Error selecting year")
    
    print(f"Debug - Using year: {year_label}, data type: {type(year_data)}")
    
    if year_data is None:
        return _create_error_table("No data available for selected year")
        
    if hasattr(year_data, 'empty') and year_data.empty:
        return _create_error_table("No data available for selected year")
    
    # Handle both Series and DataFrame
    if isinstance(year_data, pd.DataFrame):
        print(f"Debug - DataFrame columns: {year_data.columns.tolist()}")
        # Use the first numeric column or a specific column for analysis
        numeric_cols = year_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return _create_error_table("No numeric columns found in data")
        # Use the first numeric column (you may want to specify which column to use)
        year_series = year_data[numeric_cols[0]]
        print(f"Debug - Using column '{numeric_cols[0]}' for analysis")
    else:
        year_series = year_data
    
    # Ensure index is datetime
    if not isinstance(year_series.index, pd.DatetimeIndex):
        try:
            year_series.index = pd.to_datetime(year_series.index)
            print("Debug - Converted index to datetime")
        except Exception as e:
            print(f"Debug - Failed to convert index: {e}")
            return _create_error_table("Cannot convert index to datetime")
    
    # Convert series values to numeric
    try:
        year_data_numeric = pd.to_numeric(year_series, errors='coerce')
        print(f"Debug - Numeric conversion successful, shape: {year_data_numeric.shape}")
    except Exception as e:
        print(f"Debug - Numeric conversion failed: {e}")
        return _create_error_table("Cannot convert data to numeric")
    
    # Drop NaN values after conversion
    year_data_numeric = year_data_numeric.dropna()
    print(f"Debug - After dropping NaN, shape: {year_data_numeric.shape}")
    
    if year_data_numeric.empty:
        return _create_error_table("No valid numeric data for selected year")
    
    # Prepare daily statistics
    daily_stats = []
    
    # Get unique dates (limit to max_days most recent)
    try:
        unique_dates = sorted(year_data_numeric.index.date)[-max_days:]
        print(f"Debug - Processing {len(unique_dates)} unique dates")
    except Exception as e:
        print(f"Debug - Error getting unique dates: {e}")
        return _create_error_table("Error processing dates")
    
    for date in unique_dates:
        try:
            date_data = year_data_numeric[year_data_numeric.index.date == date]
            if not date_data.empty:
                stats = {
                    'Date': date.strftime('%Y-%m-%d'),
                    'High': date_data.max(),
                    'Low': date_data.min(),
                    'Open': date_data.iloc[0],
                    'Close': date_data.iloc[-1],
                    'Mean': date_data.mean(),
                    'StDev': date_data.std() if len(date_data) > 1 else 0.0,
                    'Count': len(date_data)
                }
                daily_stats.append(stats)
        except Exception as e:
            print(f"Debug - Error processing date {date}: {e}")
            continue
    
    print(f"Debug - Created {len(daily_stats)} daily statistics")
    
    if not daily_stats:
        return _create_error_table("No daily data available")
    
    # Convert to DataFrame for easier manipulation
    try:
        stats_df = pd.DataFrame(daily_stats)
        print(f"Debug - Stats DataFrame shape: {stats_df.shape}")
    except Exception as e:
        print(f"Debug - Error creating stats DataFrame: {e}")
        return _create_error_table("Error creating statistics table")
    
    # Prepare table data
    try:
        header_values = ['Date', 'High', 'Low', 'Open', 'Close', 'Mean', 'StDev', 'Count']
        
        cell_values = []
        for _, row in stats_df.iterrows():
            cell_values.append([
                row['Date'],
                f"{row['High']:.2f}",
                f"{row['Low']:.2f}",
                f"{row['Open']:.2f}",
                f"{row['Close']:.2f}",
                f"{row['Mean']:.2f}",
                f"{row['StDev']:.2f}",
                f"{int(row['Count'])}"
            ])
        
        # Transpose for correct table layout
        cell_values_transposed = list(zip(*cell_values))
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header_values,
                fill_color='lightturquoise',
                align='center',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=cell_values_transposed,
                fill_color='lavender',
                align='center',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title=f"Daily Summary Statistics - {year_label}",
            height=400,
            template='plotly_white'
        )
        
        print("Debug - Table created successfully")
        return fig
        
    except Exception as e:
        print(f"Debug - Error creating table: {e}")
        return _create_error_table(f"Error creating table: {str(e)}")


def _create_error_table(message):
    """Helper function to create an error table"""
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Error'], fill_color='lightcoral'),
        cells=dict(values=[[message]], fill_color='mistyrose')
    )])
    fig.update_layout(height=200, template='plotly_white')
    return fig


# Additional helper function to show DataFrame structure for debugging
def debug_data_structure(seasonal_data, year_key=None):
    """
    Helper function to debug the structure of your seasonal data
    """
    if not seasonal_data:
        print("No seasonal data provided")
        return
    
    for key, data in seasonal_data.items():
        if year_key and key != year_key:
            continue
            
        print(f"\n=== Year: {key} ===")
        print(f"Type: {type(data)}")
        
        if data is None:
            print("Data is None")
            continue
            
        if hasattr(data, 'shape'):
            print(f"Shape: {data.shape}")
            
        if hasattr(data, 'columns'):
            print(f"Columns: {data.columns.tolist()}")
            print(f"Data types: {data.dtypes.to_dict()}")
            
        if hasattr(data, 'index'):
            print(f"Index type: {type(data.index)}")
            if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                try:
                    print(f"Index range: {data.index.min()} to {data.index.max()}")
                except:
                    print("Could not determine index range")
                    
        # Show first few rows
        if hasattr(data, 'head'):
            print("First few rows:")
            print(data.head())


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime as dt # Added for latest_year determination

def plot_daily_pnl_with_trades(equity_curves, trade_details, position_size=1, contract_multiplier=100):
    """
    Plot daily PnL in dollar terms with trade entry/exit points.
    
    Parameters:
    - equity_curves: dict of {year: pd.Series} with daily PnL values
    - trade_details: dict of {year: trade_info_dict}
    - position_size: number of contracts
    - contract_multiplier: dollar value per point
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily PnL Development', 'Trade Entry/Exit Timeline'),
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25]
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Determine the latest year for bold black line
    latest_year = max(equity_curves.keys()) if equity_curves else None

    # Plot daily PnL curves
    for i, (year, curve) in enumerate(equity_curves.items()):
        final_pnl = curve.iloc[-1] if len(curve) > 0 else 0
        
        # Apply bold black line to the latest year, otherwise use color based on PnL
        if year == latest_year:
            line_color = 'white'
            line_width = 3
        else:
            line_color = '#2E8B57' if final_pnl > 0 else '#DC143C'
            line_width = 2
            
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve.values,
                mode='lines',
                name=f'{year} (${final_pnl:,.0f})',
                line=dict(color=line_color, width=line_width),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Date: %{x|%Y-%m-%d}<br>' +
                                  'PnL: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )

    # Plot trade markers
    for i, (year, details) in enumerate(trade_details.items()):
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[pd.to_datetime(details['entry_date'])],
                y=[0],
                mode='markers',
                marker=dict(color='green', size=12, symbol='triangle-up'),
                name=f'{year} Entry',
                hovertemplate=f'<b>{year} Entry</b><br>' +
                                  f'Date: {details["entry_date"]}<br>' +
                                  f'Price: {details["entry_price"]:.2f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Exit marker
        fig.add_trace(
            go.Scatter(
                x=[pd.to_datetime(details['exit_date'])],
                y=[details.get('dollar_pnl', 0)],
                mode='markers',
                marker=dict(color='red', size=12, symbol='triangle-down'),
                name=f'{year} Exit',
                hovertemplate=f'<b>{year} Exit</b><br>' +
                                  f'Date: {details["exit_date"]}<br>' +
                                  f'Price: {details["exit_price"]:.2f}<br>' +
                                  f'PnL: ${details.get("dollar_pnl", 0):,.2f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )

    # Add zero lines
    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=2, col=1)

    fig.update_layout(
        title='Buy & Hold Strategy - Daily PnL Analysis',
        height=700,
        template='plotly_white', # Already aligned
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Trade PnL ($)", row=2, col=1)

    return fig
def plot_pnl_distribution(returns_df):
    """
    Plot PnL distribution histogram with statistics.
    """
    fig = go.Figure()
    
    avg_pnl = returns_df['Dollar PnL'].mean()
    
    fig.add_trace(go.Histogram(
        x=returns_df['Dollar PnL'],
        nbinsx=min(20, len(returns_df)),
        marker_color='lightblue',
        opacity=0.7,
        name='PnL Distribution'
    ))

    # Add statistics lines
    fig.add_vline(
        x=avg_pnl,
        line=dict(color='red', width=2, dash='dash'),
        annotation_text=f'Mean: ${avg_pnl:,.0f}',
        annotation_position='top right'
    )
    
    fig.add_vline(
        x=returns_df['Dollar PnL'].median(),
        line=dict(color='orange', width=2, dash='dash'),
        annotation_text=f'Median: ${returns_df["Dollar PnL"].median():,.0f}',
        annotation_position='top left'
    )

    fig.update_layout(
        title='Distribution of Trade PnL',
        xaxis_title='PnL ($)',
        yaxis_title='Frequency',
        bargap=0.1,
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_monthly_pnl_drawdown(equity_curves):
    """
    Create monthly PnL and drawdown analysis similar to seasonal charts.
    Months on x-axis, years as different lines.
    
    Parameters:
    - equity_curves: dict of {year: pd.Series} with daily PnL values
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly PnL by Year', 'Monthly Maximum Drawdown by Year'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    monthly_pnl_data = {}
    monthly_dd_data = {}
    
    # Process each year's data
    for year, curve in equity_curves.items():
        if len(curve) == 0:
            continue
            
        # Ensure datetime index
        if not isinstance(curve.index, pd.DatetimeIndex):
            curve.index = pd.to_datetime(curve.index)
        
        # Calculate monthly PnL (change from month to month)
        monthly_curve = curve.groupby(curve.index.month).last()
        monthly_pnl = monthly_curve.diff().fillna(monthly_curve.iloc[0] if len(monthly_curve) > 0 else 0)
        
        # Calculate drawdown
        running_max = curve.expanding().max()
        drawdown = curve - running_max
        monthly_dd = drawdown.groupby(drawdown.index.month).min()  # Worst drawdown in each month
        
        monthly_pnl_data[year] = monthly_pnl
        monthly_dd_data[year] = monthly_dd
    
    # Determine the latest year
    latest_year = max(equity_curves.keys()) if equity_curves else None

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot monthly PnL
    color_idx = 0
    for year, monthly_pnl in monthly_pnl_data.items():
        line_color = colors[color_idx % len(colors)]
        line_width = 2
        line_dash = 'solid'

        if year == latest_year:
            line_color = 'white'
            line_width = 4
            line_dash = 'solid'
        else:
            color_idx += 1 # Only increment for non-latest years to use distinct colors

        # Ensure we have data for all 12 months
        full_monthly_pnl = pd.Series(index=range(1, 13), dtype=float)
        for month in range(1, 13):
            if month in monthly_pnl.index:
                full_monthly_pnl[month] = monthly_pnl.loc[month]
            else:
                full_monthly_pnl[month] = 0
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=full_monthly_pnl.values,
                mode='lines+markers',
                name=f'{year}',
                line=dict(color=line_color, width=line_width, dash=line_dash),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Month: %{x}<br>' +
                                'PnL: $%{y:,.2f}<extra></extra>',
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Plot monthly drawdown
    color_idx = 0 # Reset color index for drawdown plot
    for year, monthly_dd in monthly_dd_data.items():
        line_color = colors[color_idx % len(colors)]
        line_width = 2
        line_dash = 'dash'

        if year == latest_year:
            line_color = 'black'
            line_width = 4
            line_dash = 'solid'
        else:
            color_idx += 1 # Only increment for non-latest years to use distinct colors

        # Ensure we have data for all 12 months
        full_monthly_dd = pd.Series(index=range(1, 13), dtype=float)
        for month in range(1, 13):
            if month in monthly_dd.index:
                full_monthly_dd[month] = monthly_dd.loc[month]
            else:
                full_monthly_dd[month] = 0
        
        fig.add_trace(
            go.Scatter(
                x=months,
                y=full_monthly_dd.values,
                mode='lines+markers',
                name=f'{year}',
                line=dict(color=line_color, width=line_width, dash=line_dash),
                marker=dict(size=6),
                hovertemplate='<b>%{fullData.name} DD</b><br>' +
                                'Month: %{x}<br>' +
                                'Max DD: $%{y:,.2f}<extra></extra>',
                showlegend=False # Drawdown lines share legend with PnL lines if both shown
            ),
            row=2, col=1
        )
    
    # Remove average lines as per user request
    # if len(monthly_pnl_data) > 1:
    #     # Calculate average monthly PnL
    #     all_monthly_pnl = pd.DataFrame(monthly_pnl_data)
    #     avg_monthly_pnl = all_monthly_pnl.mean(axis=1)
        
    #     # Ensure 12 months
    #     full_avg_pnl = pd.Series(index=range(1, 13), dtype=float)
    #     for month in range(1, 13):
    #         if month in avg_monthly_pnl.index:
    #             full_avg_pnl[month] = avg_monthly_pnl.loc[month]
    #         else:
    #             full_avg_pnl[month] = 0
        
    #     fig.add_trace(
    #         go.Scatter(
    #             x=months,
    #             y=full_avg_pnl.values,
    #             mode='lines+markers',
    #             name='Average PnL',
    #             line=dict(color='black', width=4, dash='solid'),
    #             marker=dict(size=8, color='black'),
    #             hovertemplate='<b>Average PnL</b><br>' +
    #                           'Month: %{x}<br>' +
    #                           'Avg PnL: $%{y:,.2f}<extra></extra>'
    #         ),
    #         row=1, col=1
    #     )
        
    #     # Calculate average monthly drawdown
    #     all_monthly_dd = pd.DataFrame(monthly_dd_data)
    #     avg_monthly_dd = all_monthly_dd.mean(axis=1)
        
    #     # Ensure 12 months
    #     full_avg_dd = pd.Series(index=range(1, 13), dtype=float)
    #     for month in range(1, 13):
    #         if month in avg_monthly_dd.index:
    #             full_avg_dd[month] = avg_monthly_dd.loc[month]
    #         else:
    #             full_avg_dd[month] = 0
        
    #     fig.add_trace(
    #         go.Scatter(
    #             x=months,
    #             y=full_avg_dd.values,
    #             mode='lines+markers',
    #             name='Average DD',
    #             line=dict(color='red', width=4, dash='dash'),
    #             marker=dict(size=8, color='red'),
    #             hovertemplate='<b>Average Drawdown</b><br>' +
    #                           'Month: %{x}<br>' +
    #                           'Avg DD: $%{y:,.2f}<extra></extra>',
    #             showlegend=False
    #         ),
    #         row=2, col=1
    #     )
    
    # Add zero lines
    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=2, col=1)
    
    fig.update_layout(
        title='Monthly PnL & Drawdown Analysis',
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Monthly PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Max Drawdown ($)", row=2, col=1)
    
    return fig

def plot_performance_metrics(returns_df):
    """
    Plot key performance metrics in a dashboard style.
    """
    profitable_trades = returns_df[returns_df['Dollar PnL'] > 0]
    losing_trades = returns_df[returns_df['Dollar PnL'] < 0]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Win/Loss Distribution',
            'PnL by Year',
            'Days Held Distribution',
            'Cumulative Performance'
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )

    # Win/Loss pie chart
    win_count = len(profitable_trades)
    loss_count = len(losing_trades)
    
    fig.add_trace(
        go.Pie(
            labels=['Wins', 'Losses'],
            values=[win_count, loss_count],
            marker_colors=['#2E8B57', '#DC143C'],
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        ),
        row=1, col=1
    )

    # PnL by year bar chart
    fig.add_trace(
        go.Bar(
            x=returns_df['Year'],
            y=returns_df['Dollar PnL'],
            marker_color=['#2E8B57' if pnl > 0 else '#DC143C' for pnl in returns_df['Dollar PnL']],
            hovertemplate='<b>Year %{x}</b><br>PnL: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Days held histogram
    fig.add_trace(
        go.Histogram(
            x=returns_df['Days Held'],
            nbinsx=15,
            marker_color='skyblue',
            opacity=0.7,
            hovertemplate='Days: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # Cumulative performance
    cumulative_pnl = returns_df.sort_values('Year')['Dollar PnL'].cumsum()
    fig.add_trace(
        go.Scatter(
            x=returns_df.sort_values('Year')['Year'],
            y=cumulative_pnl,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Year %{x}</b><br>Cumulative PnL: $%{y:,.2f}<extra></extra>'
        ),
        row=2, col=2
    )

    fig.update_layout(
        title='Performance Dashboard',
        height=600,
        template='plotly_white',
        showlegend=False
    )

    return fig