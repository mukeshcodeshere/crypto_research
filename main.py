# Once you install the dependencies in requirements.txt, you can use "python main.py -c /path/to/venue_defs.yml -e coinbase" to start the process. Take a look at the script for some additional parameters if you want, such as specifying the exact instruments you want to pull and the data directory to install to. If you pull for all venue instruments in the file, the process will take several hours, so you can also limit it to just BTCUSD initially if you want
# Import required libraries
import argparse  # For parsing command-line arguments
from datetime import datetime  # For handling timestamps
import glob  # For finding files using Unix-style pathname patterns
import logging  # For logging messages
import os  # For interacting with the filesystem
import sys  # For accessing system-specific parameters
import time  # For time-related functions

# Data handling libraries
import pandas as pd
import numpy as np

# Rate limiting session for HTTP requests
from requests_ratelimiter import LimiterSession

# YAML file parser
import yaml

# Handle timezone differently depending on Python version
if sys.version_info < (3, 11, 0):
    from datetime import timezone
    UTC = timezone.utc
else:
    from datetime import UTC

# Set up logging format
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define constant for 5-minute candle granularity
SECONDS = 300


# Instrument class to store symbol and latest timestamp
class Instrument:
    def __init__(self, symbol, venue_symbol):
        self.symbol = symbol  # Standardized internal symbol (e.g., BTCUSD)
        self.venue_symbol = venue_symbol  # Exchange-specific symbol (e.g., BTC-USD)
        self.latest_timestamp = 0  # Last known timestamp for which data exists


# Coinbase exchange handler
class Coinbase:
    def __init__(self):
        # Use rate-limited session to avoid hitting API limits
        self.client = LimiterSession(per_second=5)
        self.request_limit = 300  # Coinbase allows up to 300 candles per request

    def get_candles(self, now: int, instrument: 'Instrument', dfs) -> bool:
        # Prepare parameters for API call
        params = {
            'granularity': SECONDS,
            'end': now - (now % SECONDS) - SECONDS  # Align to last full 5-min block
        }

        processing = True

        while processing:
            # Calculate start time based on latest timestamp or API limits
            params['start'] = max(instrument.latest_timestamp, params['end'] - self.request_limit * SECONDS)

            logging.info(f"retrieving {instrument.symbol} candles from {to_datetime(params['start'])} to {to_datetime(params['end'])}")
            r = self.client.get(f'https://api.exchange.coinbase.com/products/{instrument.venue_symbol}/candles', params=params)

            # Handle HTTP errors
            if r.status_code != 200:
                if r.status_code == 429:
                    # Rate limit exceeded; wait and retry
                    logging.error(f'rate limit exceeded for {instrument.symbol}, delaying and retrying {r.text}')
                    time.sleep(1)
                    continue
                else:
                    # Other errors; stop processing
                    processing = False
                    logging.error(f'failed to get candles for {instrument.symbol} {r.status_code} {r.text}')
                    break

            data = r.json()
            if not data:
                # No more data to pull
                break

            # Reverse to chronological order
            df = pd.DataFrame(data[::-1], columns=['time','low','high','open','close','volume'])

            # Fill in missing 5-minute intervals with NaN
            df = df.set_index('time')
            df = df.reindex(np.arange(params['start'], params['end'] + 1, SECONDS)).reset_index()

            logging.info(f"{instrument.symbol} had {len(df)} records from {to_datetime(params['start'])} to {to_datetime(params['end'])}")

            dfs.append(df)

            # Pagination: move the window backwards
            if params['start'] > instrument.latest_timestamp:
                params['end'] = params['start'] - SECONDS
            else:
                break

        return processing


# Binance exchange handler
class Binance:
    def __init__(self):
        self.client = LimiterSession(per_second=5)
        self.request_limit = 1000  # Binance allows up to 1000 candles per request

    def get_candles(self, now: int, instrument: 'Instrument', dfs) -> bool:
        params = {
            'symbol': instrument.venue_symbol,
            'interval': '5m',
            'endTime': (now - (now % SECONDS) - SECONDS) * 1000,
            'limit': self.request_limit
        }

        processing = True

        while processing:
            params['startTime'] = max(instrument.latest_timestamp * 1000, params['endTime'] - params['limit'] * SECONDS * 1000)

            logging.info(f'retrieving {instrument.symbol} candles from {to_datetime(params["startTime"]/1000)} to {to_datetime(params["endTime"]/1000)}')
            r = self.client.get('https://data-api.binance.vision/api/v3/klines', params=params)

            if r.status_code != 200:
                if r.status_code == 429:
                    logging.error(f'rate limit exceeded for {instrument.symbol}, delaying and retrying {r.text}')
                    time.sleep(1)
                    continue
                else:
                    processing = False
                    logging.error(f'failed to get candles for {instrument.symbol} {r.status_code} {r.text}')
                    break

            data = r.json()
            if not data:
                break

            # Extract relevant columns and format
            df = pd.DataFrame(data, columns=[
                'time','open','high','low','close','volume','_closetime',
                '_quotevolume','_trades','_takerbuybasevol','_takerbuyquotevol','_ignore'
            ]).astype({
                'time': 'int',
                'open': 'float',
                'high': 'float',
                'low': 'float',
                'close': 'float',
                'volume': 'float'
            })

            # Convert from ms to seconds
            df['time'] = (df['time'] / 1000).astype(int)

            # Fill in missing time intervals
            df = df.set_index('time')
            df = df.reindex(np.arange(int(params['startTime'] / 1000), int(params['endTime'] / 1000 - SECONDS) + 1, SECONDS)).reset_index()

            logging.info(f"{instrument.symbol} had {len(df)} records from {to_datetime(params['startTime']/1000)} to {to_datetime(params['endTime']/1000)}")

            dfs.append(df[['time','open','high','low','close','volume']])

            # Pagination logic
            if params['startTime'] > instrument.latest_timestamp * 1000:
                params['endTime'] = params['startTime']
            else:
                break

        return processing


# Validate that the provided file path exists
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error(f'file {arg} does not exist')
    else:
        return open(arg, 'r')  # Return file object


# Convert UNIX timestamp to human-readable string
def to_datetime(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime('%Y-%m-%d %H:%M:%S')


# Main function that orchestrates the script
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='config file with venue instruments',
                        metavar='FILE', type=lambda x: is_valid_file(parser, x))
    parser.add_argument('-d', '--dryrun', action='store_true', help='dry run without persisting data')
    parser.add_argument('-e', '--exchange', required=True, choices=['binance', 'coinbase'], help='exchange identifier')
    parser.add_argument('-i', '--instruments', type=str, help='comma delimited list of instruments to process')
    parser.add_argument('--data-dir', default=None, help='path to data directory')  # <-- set default to None
    args = parser.parse_args()

    if args.data_dir:
        base_dir = args.data_dir
    else:
        base_dir = os.getcwd()  # <-- current working directory

    data_dir = os.path.join(base_dir, args.exchange)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logging.info(f'Created missing data directory at {data_dir}')
    
    # Select the appropriate adapter for the chosen exchange
    adapter = Coinbase() if args.exchange == 'coinbase' else Binance()

    filters = {}
    if args.instruments:
        filters = {i: None for i in args.instruments.split(',')}

    instruments = []

    # Load config and parse instruments
    with args.config as f:
        config = yaml.safe_load(f)
        vd = [v for v in config['venue_defs'] if v['name'] == args.exchange]
        if not vd:
            logging.error(f'failed to find venue instrument config for {args.exchange}')
            exit(1)

        for inst in vd[0]['instruments']:
            symbol = inst['symbol'].replace('/', '')
            if filters and symbol not in filters:
                continue
            instruments.append(Instrument(symbol, inst['venue_symbol']))

    # Determine latest timestamp for each instrument by scanning local CSV files
    for inst in instruments:
        pattern = f'{data_dir}/5m/*/{inst.symbol}_5m_*.csv'
        matches = glob.glob(pattern, recursive=True)
        if matches:
            for match in matches:
                with open(match, 'r') as f:
                    line = f.readlines()[-1]
                    ts = int(line.strip().split(',')[0])
                    if ts > inst.latest_timestamp:
                        inst.latest_timestamp = ts
        else:
            logging.warning(f'no matches found for {inst.symbol}, data pull will start from beginning')

    now = int(time.time())  # Current time in UNIX timestamp

    # For each instrument, pull and store data
    for inst in instruments:
        dfs = []
        success = adapter.get_candles(now, inst, dfs)

        if dfs and success:
            df = pd.concat(dfs, ignore_index=True)
            df.sort_values(by='time', ignore_index=True, inplace=True)

            # Adjust candle timestamp to reflect closing time
            df['time'] = df['time'] + SECONDS

            logging.info(f'retrieved {len(df)} new records for {inst.symbol}')

            if args.dryrun:
                continue

            # Save data grouped by year
            for year, dff in df.groupby(pd.to_datetime(df['time'], unit='s').dt.year):
                outdir = os.path.expanduser(f'{data_dir}/5m/{year}')
                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                dest = f'{outdir}/{inst.symbol}_5m_{year}.csv'
                logging.info(f'saving {len(dff)} records to {dest}')

                dff[['time','open','high','low','close','volume']].to_csv(
                    dest, mode='a', index=False, header=False, na_rep='NaN'
                )

    logging.info(f'completed update for {len(instruments)} instruments')


# Entry point of script
if __name__ == "__main__":
    main()
