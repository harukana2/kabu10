"""
kabu_common.py
------------------------------------------------------------
データ取得・指標計算・パターン判定・フィルタ処理を一元化したモジュール。
kabu_screener_v2.py はここの関数を通して判定を行う。

■ 本バージョンの方針
指定された5つのハードフィルタ／パターン条件と、任意でON/OFFできる
RSI・ボリンジャーバンド・MACDの3指標のみを扱う。
旧版にあったスコアリング・約定しやすさ・バックテスト較正・
近傍補完などのロジックは削除した。
------------------------------------------------------------
"""

import math
import time

import numpy as np
import pandas as pd

import kabu_config as cfg

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# ============================================================
# データ取得
# ============================================================

def _normalize_columns(data, symbols):
    """
    銘柄数が1件のみのチャンクだと yfinance がフラットな列(MultiIndexでない)で
    返してくることがあるため、常に (フィールド, ティッカー) の2階層に揃える。
    """
    if not isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = pd.MultiIndex.from_product([data.columns, symbols])
    return data


def download_bulk_data(symbols, period, interval="1d", chunk_size=cfg.CHUNK_SIZE,
                        chunk_sleep_sec=cfg.CHUNK_SLEEP_SEC, verbose=True):
    """複数銘柄の日足データをチャンク分割して一括取得する。"""
    if not HAS_YFINANCE:
        raise RuntimeError("yfinance がインストールされていません。`pip install yfinance` を実行してください。")

    total_chunks = math.ceil(len(symbols) / chunk_size)
    frames = []

    if verbose:
        print(f"{len(symbols)} 銘柄の株価データを {total_chunks} チャンクに分けて取得中... (period={period})")

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i: i + chunk_size]
        chunk_no = i // chunk_size + 1
        if verbose:
            print(f"  [{chunk_no}/{total_chunks}] {len(chunk)}銘柄取得中...")
        try:
            d = yf.download(
                chunk, period=period, interval=interval, progress=False,
                group_by="column", auto_adjust=False, threads=True,
            )
            if d is not None and not d.empty:
                frames.append(_normalize_columns(d, chunk))
        except Exception as e:
            if verbose:
                print(f"  [{chunk_no}/{total_chunks}] 取得に失敗しました: {e}")
        if chunk_no < total_chunks:
            time.sleep(chunk_sleep_sec)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def get_all_japanese_tickers():
    """JPXから全上場銘柄のリストを取得する（プライム/スタンダード/グロース）"""
    print("JPXから全銘柄リストを取得中...")
    try:
        df = pd.read_excel(cfg.JPX_LIST_URL)
        df = df[df["市場・商品区分"].str.contains(cfg.TARGET_MARKETS_REGEX, na=False)]
        tickers = df["コード"].astype(str).tolist()
        print(f"  対象銘柄数: {len(tickers)}")
        return tickers
    except Exception as e:
        print(f"銘柄リストの取得に失敗しました: {e}")
        return []


def parse_ticker_code(code_val):
    """銘柄コードの整形（末尾の '.0' '.T' 等を除去）"""
    if pd.isna(code_val):
        return None
    s = str(code_val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.lower().endswith(".t"):
        s = s[:-2]
    return s if s else None


# ============================================================
# 指標計算
# 引数の close/open/high/low/volume は index=日付, columns=銘柄コード
# の2次元DataFrameを想定
# ============================================================

def calc_rsi(close, window=cfg.RSI_WINDOW, method=cfg.RSI_METHOD):
    """
    RSIを計算する。
    method="wilder": Wilderの指数平滑（標準的なチャートツールと同じ方式）
    method="sma"   : 単純移動平均ベース
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    if method == "wilder":
        avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    else:
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close, fast=cfg.MACD_FAST, slow=cfg.MACD_SLOW, signal=cfg.MACD_SIGNAL):
    """MACD本体・シグナル線・ヒストグラムを返す"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calc_bollinger(close, window=cfg.BB_WINDOW, num_std=cfg.BB_NUM_STD):
    """ボリンジャーバンド（中心線・上限・下限）を返す"""
    mid = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = mid + std * num_std
    lower = mid - std * num_std
    return mid, upper, lower


def compute_all_indicators(closes, opens, highs, lows, volumes):
    """
    一括ダウンロードした OHLCV(DataFrame, columns=銘柄) から、
    スクリーニングに必要な指標一式をまとめて計算する。
    戻り値は dict[str, DataFrame] で、各指標の時系列全体を保持する
    （前日・前々日の値を参照するため）。
    """
    mid_ma = closes.rolling(window=cfg.MID_TERM_MA_WINDOW).mean()
    avg_volume = volumes.rolling(window=cfg.VOLUME_MA_WINDOW).mean()

    rsi = calc_rsi(closes)
    macd, macd_signal, macd_hist = calc_macd(closes)
    bb_mid, bb_upper, bb_lower = calc_bollinger(closes)

    return {
        "close": closes, "open": opens, "high": highs, "low": lows, "volume": volumes,
        "mid_ma": mid_ma, "avg_volume": avg_volume,
        "rsi": rsi, "macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "bb_mid": bb_mid, "bb_upper": bb_upper, "bb_lower": bb_lower,
    }


# ============================================================
# ハードフィルタ（単元価格上限・出来高下限）
# ============================================================

def apply_hard_filter_at(ind, date_idx):
    """
    指標一式(ind)のうち date_idx 時点のスナップショットに対して
    ハードフィルタ（単元価格上限・出来高下限）を適用し、
    通過した銘柄のブール Series を返す。
    """
    latest_close = ind["close"].loc[date_idx]
    avg_volume = ind["avg_volume"].loc[date_idx]

    cond_price = (latest_close * cfg.UNIT_SHARES) <= cfg.MAX_UNIT_PRICE
    cond_volume = avg_volume >= cfg.MIN_AVG_VOLUME

    final_cond = cond_price & cond_volume
    return final_cond.fillna(False)


# ============================================================
# ローソク足パターン判定
# ------------------------------------------------------------
# date_idx を「当日(陽線で中期線を回復した日)」として、
#   前々日: 終値が中期線の上
#   前日  : 陰線 かつ 終値が中期線を割り込んでいる
#   当日  : 陽線 かつ 終値が中期線を上回って戻している
#   当日の始値が前日(陰線)の終値より上（ギャップアップで寄り付き）
# の5条件すべてを満たす銘柄を抽出する。
# ============================================================

def apply_pattern_filter_at(ind, date_idx):
    close = ind["close"]
    open_ = ind["open"]
    mid_ma = ind["mid_ma"]

    close_0 = close.loc[date_idx]
    open_0 = open_.loc[date_idx]
    mid_0 = mid_ma.loc[date_idx]

    close_1 = close.shift(1).loc[date_idx]     # 前日の終値
    open_1 = open_.shift(1).loc[date_idx]      # 前日の始値
    mid_1 = mid_ma.shift(1).loc[date_idx]      # 前日時点の中期線

    close_2 = close.shift(2).loc[date_idx]     # 前々日の終値
    mid_2 = mid_ma.shift(2).loc[date_idx]      # 前々日時点の中期線

    # 条件5: 前々日の終値が中期線の上
    cond_day_minus2_above = close_2 > mid_2

    # 条件3(前半): 前日は陰線 かつ 中期線を割り込んでいる
    cond_day_minus1_bearish = close_1 < open_1
    cond_day_minus1_below_mid = close_1 < mid_1

    # 条件3(後半): 当日は陽線 かつ 中期線を上回って戻している
    cond_day0_bullish = close_0 > open_0
    cond_day0_above_mid = close_0 > mid_0

    # 条件4: 当日の始値が前日(陰線)の終値より上（ギャップアップ）
    cond_gap_up_open = open_0 > close_1

    pattern = (
        cond_day_minus2_above
        & cond_day_minus1_bearish
        & cond_day_minus1_below_mid
        & cond_day0_bullish
        & cond_day0_above_mid
        & cond_gap_up_open
    )
    return pattern.fillna(False)


# ============================================================
# 追加のテクニカル指標フィルタ（RSI / ボリンジャーバンド / MACD）
# それぞれ kabu_config.py の ENABLE_xxx_FILTER で ON/OFF できる。
# ============================================================

def apply_indicator_filters_at(ind, date_idx):
    """
    RSI・ボリンジャーバンド・MACDによる追加フィルタを、ON になっている
    ものだけ適用する。すべてOFFの場合は全銘柄 True（素通り）になる。
    """
    symbols = ind["close"].columns
    cond = pd.Series(True, index=symbols)

    if cfg.ENABLE_RSI_FILTER:
        rsi_0 = ind["rsi"].loc[date_idx]
        cond &= (rsi_0 >= cfg.RSI_MIN) & (rsi_0 <= cfg.RSI_MAX)

    if cfg.ENABLE_BB_FILTER and cfg.BB_REQUIRE_PREV_LOW_TOUCH_LOWER:
        low_1 = ind["low"].shift(1).loc[date_idx]
        bb_lower_1 = ind["bb_lower"].shift(1).loc[date_idx]
        cond &= (low_1 <= bb_lower_1)

    if cfg.ENABLE_MACD_FILTER and cfg.MACD_REQUIRE_HIST_RISING:
        hist_0 = ind["macd_hist"].loc[date_idx]
        hist_1 = ind["macd_hist"].shift(1).loc[date_idx]
        cond &= (hist_0 > hist_1)

    return cond.fillna(False)
