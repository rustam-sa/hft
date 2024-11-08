timeframes_in_minutes = {
     "1min": 1,
     "3min": 3,
     "5min": 5, 
    "15min": 15,
    "30min": 30,
    "1hour": 60,
    "2hour": 120,
    "4hour": 240,
    "6hour": 360,
    "8hour": 480,
   "12hour": 720,
     "1day": 1440,
    "1week": 10080
    }

kucoin_endpoints = {
    "base": "https://api.kucoin.com",
    "accounts": "/api/v1/accounts",
    "transferable": "/api/v1/accounts/transferable",
    "orders": "/api/v1/orders",
    "symbols": "/api/v1/symbols",
    "candles": "/api/v1/market/candles",
    "websocket_info": "/api/v1/bullet-public",
    "margin_info": "/api/v1/margin/config",
    "sub_accounts": "/api/v1/sub/user",
    "ledgers": "/api/v1/accounts/ledgers",
    "sub_balance": "/api/v1/sub-accounts",
    "all_sub_balance": "/api/v1/sub-accounts",
    "sub_transfer": "/api/v2/accounts/sub-transfer",
    "inner_transfer": "/api/v2/accounts/inner-transfer",
    "create_deposit_address": "/api/v1/deposit-addresses",
    "get_deposit_addresses": "/api/v2/deposit-addresses",
    "get_deposit_address": "/api/v1/deposit-addresses",
    "get_deposit_list": "/api/v1/deposits",
    "get_withdrawals_list": "/api/v1/withdrawals",
    "get_withdrawal_quotas": "/api/v1/withdrawals/quotas",
    "apply_withdraw": "/api/v1/withdrawals",
    "cancel_withdrawal": "/api/v1/withdrawals",
    "base_fee": "/api/v1/base-fee",
    "trade_fee": "/api/v1/trade-fees",
    "new_order": "/api/v1/orders",
    "margin_order": "/api/v1/margin/order",
    "cancel_order": "/api/v1/orders",
    "list_orders": "/api/v1/orders",
    "get_order": "/api/v1/orders/",
    "fills": "/api/v1/fills",
    "stop_order": "/api/v1/stop-order",
    "stop_order_client_id": "/api/v1/stop-order/queryOrderByClientOid",
    "cancel_stop_order_by_client_id": "/api/v1/stop-order/cancelOrderByClientOid",
    "ticker": "/api/v1/market/orderbook/level1",
    "markets": "/api/v1/markets",
    "currencies": "/api/v2/currencies",
    "prices": "/api/v1/prices",
    "margin_config": "/api/v1/margin/config",
    "margin_account": "/api/v1/margin/account",
    "margin_borrow": "/api/v1/margin/borrow",
    "borrow_outstanding": "/api/v1/margin/borrow/outstanding",
    "borrow_repaid": "/api/v1/margin/borrow/repaid",
    "one_click_repayment": "/api/v1/margin/repay/all",
    "repay_single_order": "/api/v1/margin/repay/single",
    "lend_order": "/api/v1/margin/lend",
    "toggle_auto_lend": "/api/v1/margin/toggle-auto-lend",
    "lend_active": "/api/v1/margin/lend/active",
    "lent_history": "/api/v1/margin/lend/done",
    "active_lend_orders": "/api/v1/margin/lend/trade/unsettled",
    "settled_lend_orders": "/api/v1/margin/lend/trade/settled",
    "account_lend_record": "/api/v1/margin/lend/assets",
    "lending_market_data": "/api/v1/margin/market",
    "margin_trade_data": "/api/v1/margin/trade/last",
    "server_time": "/api/v1/timestamp",
    "server_status": "/api/v1/status",
    "historical_orders": "/api/v1/hist-orders",
    'get_symbol_info': "/api/v1/symbols",
    "isolated_accounts": "/api/v1/isolated/accounts",
    "isolated_account": "/api/v1/isolated/account",
    "isolated_margin_borrow": "/api/v1/isolated/borrow",
    "isolated_repay": "/api/v1/isolated/repay/all",
    "isolated_repayment_record": "/api/v1/isolated/borrow/outstanding"
    }