API_KEY = 'PK9QJYDRU6C2572H5G2P'
SECRET_KEY = '89dhB9ioUI2Hs4JpNo6bpDJE0uLBsa0FBP4GzdQY'
HEADERS = {
  'APCA-API-KEY-ID': API_KEY,
  'APCA-API-SECRET-KEY': SECRET_KEY
}


BASE_URL = 'https://paper-api.alpaca.markets'
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)

SOCKET = 'wss://stream.data.alpaca.markets/v2/iex'
ORDER = 'https://paper-api.alpaca.markets/v2/orders'
POSITION = 'https://paper-api.alpaca.markets/v2/positions'
WATCHLIST = 'https://paper-api.alpaca.markets/v2/watchlists/e20246c3-4e90-4bbd-95be-831652fa3c0e'
CLOCK = 'https://paper-api.alpaca.markets/v2/clock'


#{"action": "auth","key": "PK1246VMPAVL8A5PX4PG","secret": "W96ScmVE7MCbrDuyPCF99ZstwiNDkt9lGhIMEtSx"}
