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
from datetime import datetime
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import kabu_config as cfg

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# ============================================================
# 取引時間チェック
# ============================================================

def is_market_open_now():
    """
    現在時刻が取引時間内かどうかを判定する。
    kabu_config.ENABLE_MARKET_HOURS_CHECK が False の場合は常に True を返す
    （＝チェックせず常に実行する）。

    ※日本の祝日（取引所の休場日）はカレンダーで判定していないため、
      祝日も「平日」として True になる点に注意。
      正確に判定したい場合は `pip install jpholiday` の上、
      この関数内で jpholiday.is_holiday(now.date()) を追加でチェックすること。
    """
    if not getattr(cfg, "ENABLE_MARKET_HOURS_CHECK", True):
        return True

    tz = ZoneInfo(getattr(cfg, "MARKET_TIMEZONE", "Asia/Tokyo"))
    now = datetime.now(tz)

    if now.weekday() >= 5:  # 5=土, 6=日
        return False

    now_t = now.time()
    for start_str, end_str in getattr(cfg, "MARKET_SESSIONS", []):
        start_t = dtime.fromisoformat(start_str)
        end_t = dtime.fromisoformat(end_str)
        if start_t <= now_t <= end_t:
            return True
    return False


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


def get_japanese_ticker_master():
    """
    JPXから全上場銘柄の一覧（プライム/スタンダード/グロース）を、
    コードと銘柄名の対応がわかる形で取得する。
    戻り値: DataFrame（列: "コード", "銘柄名"）
    """
    print("JPXから全銘柄リストを取得中...")
    try:
        df = pd.read_excel(cfg.JPX_LIST_URL)
        df = df[df["市場・商品区分"].str.contains(cfg.TARGET_MARKETS_REGEX, na=False)]
        df = df[["コード", "銘柄名"]].copy()
        df["コード"] = df["コード"].astype(str)
        print(f"  対象銘柄数: {len(df)}")
        return df
    except Exception as e:
        print(f"銘柄リストの取得に失敗しました: {e}")
        return pd.DataFrame(columns=["コード", "銘柄名"])


def get_all_japanese_tickers():
    """
    JPXから全上場銘柄のコードのみを取得する（従来互換用）。
    銘柄名も併せて使いたい場合は get_japanese_ticker_master() を使うこと。
    """
    return get_japanese_ticker_master()["コード"].tolist()


def fetch_latest_quotes(symbols, verbose=True):
    """
    指定銘柄について、直近の分足データから実勢に近い「現在値」を取得し直す。

    ■ なぜ必要か
    screen_stocks() で使っている日足データ(period=cfg.HIST_PERIOD, interval="1d")の
    「当日」の終値は、取引時間中は「その時点までの最新値」を反映するものの、
    Yahoo Finance側の更新間隔・遅延（無料データは十数分程度遅れることがある）の
    影響を受ける。10分おきなど短い間隔で実行する場合は、
    候補として絞り込んだ銘柄（数が少ない）だけ改めて1分足を取得し直すことで、
    より実勢に近い「現在値」を得られるようにしている。

    戻り値: dict[symbol] = {"price": float, "time": pd.Timestamp}
            取得できなかった銘柄はキーに含まれない（呼び出し側で日足の値に
            フォールバックすること）。
    """
    if not symbols or not HAS_YFINANCE:
        return {}

    symbols = list(symbols)
    try:
        d = yf.download(
            symbols, period="1d", interval="1m", progress=False,
            group_by="column", auto_adjust=False, threads=True,
        )
    except Exception as e:
        if verbose:
            print(f"[fetch_latest_quotes] 直近値の再取得に失敗しました: {e}")
        return {}

    if d is None or d.empty:
        return {}

    d = _normalize_columns(d, symbols)
    if "Close" not in d.columns.get_level_values(0):
        return {}
    closes = d["Close"]

    quotes = {}
    for symbol in symbols:
        if symbol not in closes.columns:
            continue
        series = closes[symbol].dropna()
        if series.empty:
            continue
        quotes[symbol] = {"price": float(series.iloc[-1]), "time": series.index[-1]}
    return quotes


# ============================================================
# テクニカル指標からの簡易評価コメント
# ------------------------------------------------------------
# RSI・MACDヒストグラム・ボリンジャーバンドの「値そのもの」は
# 見てもわかりにくいため、メールには数値の代わりに
# 「どんな地合いか」を簡潔な文章にして載せる。
# あくまで機械的なテクニカル指標の解釈であり、将来の値動きを
# 保証するものではない点に注意。
# ============================================================

def build_evaluation_text(rsi, macd_hist, macd_hist_prev, close, bb_lower, bb_upper):
    """
    RSI・MACDヒストグラム・ボリンジャーバンドの値から、
    人が読んで意味がわかる短い評価コメントを組み立てる。
    値がNoneの指標はスキップする（RSI/BB/MACDフィルタOFFでも計算自体はしているため
    値そのものは通常揃っている）。
    """
    parts = []
    bull_score = 0
    bear_score = 0

    # --- RSI: 過熱感・売られすぎ感 ---
    if rsi is not None:
        if rsi >= 70:
            parts.append(f"RSI{rsi:.0f}で過熱気味（買われすぎ水準、伸び悩みに注意）")
            bear_score += 1
        elif rsi <= 30:
            parts.append(f"RSI{rsi:.0f}で売られすぎ水準（反発が入りやすい局面）")
            bull_score += 1
        else:
            parts.append(f"RSI{rsi:.0f}で中立圏")

    # --- MACDヒストグラム: 勢いの方向 ---
    if macd_hist is not None and macd_hist_prev is not None:
        rising = macd_hist > macd_hist_prev
        positive = macd_hist > 0
        if rising and positive:
            parts.append("MACDは上昇の勢いが強まっている")
            bull_score += 1
        elif rising and not positive:
            parts.append("MACDはマイナス圏だが下落の勢いは弱まりつつある")
            bull_score += 1
        elif (not rising) and positive:
            parts.append("MACDはプラス圏だが勢いはやや鈍化")
        else:
            parts.append("MACDは下落の勢いが継続")
            bear_score += 1

    # --- ボリンジャーバンド: バンド内での位置 ---
    if bb_lower is not None and bb_upper is not None and close is not None:
        band_width = bb_upper - bb_lower
        if band_width and band_width > 0:
            position = (close - bb_lower) / band_width
            if position <= 0.15:
                parts.append("株価はBB下限付近（下げ過ぎの可能性）")
                bull_score += 1
            elif position >= 0.85:
                parts.append("株価はBB上限付近（過熱気味の可能性）")
                bear_score += 1

    if not parts:
        return "評価情報なし"

    if bull_score - bear_score >= 2:
        summary = "強気材料が優勢"
    elif bear_score - bull_score >= 2:
        summary = "弱気材料が優勢（過熱・伸び悩みに注意）"
    else:
        summary = "強気・弱気材料が拮抗（判断材料としては中立）"

    return f"{summary}／" + " ／ ".join(parts)
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
    short_ma = closes.rolling(window=getattr(cfg, "DAYTRADE_SHORT_MA_WINDOW", 5)).mean()
    avg_volume = volumes.rolling(window=cfg.VOLUME_MA_WINDOW).mean()

    rsi = calc_rsi(closes)
    macd, macd_signal, macd_hist = calc_macd(closes)
    bb_mid, bb_upper, bb_lower = calc_bollinger(closes)

    return {
        "close": closes, "open": opens, "high": highs, "low": lows, "volume": volumes,
        "mid_ma": mid_ma, "short_ma": short_ma, "avg_volume": avg_volume,
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


# ============================================================
# 残り枠の埋め合わせ（デイトレード候補）判定
# ------------------------------------------------------------
# ローソク足パターン条件（apply_pattern_filter_at）に合致しなかった
# 銘柄の中から、「本日中に上昇しやすいデイトレード向け」という
# 別の評価基準で残りの枠を埋めるために使う。
# 単元価格上限・出来高下限のハードフィルタ(apply_hard_filter_at)は
# 呼び出し側で従来通り別途適用する前提（このフィルタでは変更しない）。
#
# ここでの判定・スコアはあくまで日足データから見た機械的な目安であり、
# 実際に本日中に上昇することを保証するものではない。
# ============================================================

def apply_daytrade_filter_at(ind, date_idx):
    """
    デイトレード候補として最低限満たすべき条件を判定する。
    各条件は kabu_config.py の DAYTRADE_xxx で ON/OFF・閾値を調整できる。
    """
    close = ind["close"].loc[date_idx]
    open_ = ind["open"].loc[date_idx]
    high = ind["high"].loc[date_idx]
    low = ind["low"].loc[date_idx]
    volume = ind["volume"].loc[date_idx]
    avg_volume = ind["avg_volume"].loc[date_idx]
    mid_ma = ind["mid_ma"].loc[date_idx]
    short_ma_0 = ind["short_ma"].loc[date_idx]
    short_ma_1 = ind["short_ma"].shift(1).loc[date_idx]
    rsi = ind["rsi"].loc[date_idx]
    macd_hist_0 = ind["macd_hist"].loc[date_idx]
    macd_hist_1 = ind["macd_hist"].shift(1).loc[date_idx]

    cond = pd.Series(True, index=close.index)

    if getattr(cfg, "DAYTRADE_REQUIRE_BULLISH", True):
        cond &= (close > open_)

    if getattr(cfg, "DAYTRADE_REQUIRE_ABOVE_MID_MA", True):
        cond &= (close > mid_ma)

    if getattr(cfg, "DAYTRADE_REQUIRE_SHORT_MA_RISING", True):
        cond &= (short_ma_0 > short_ma_1)

    volume_ratio = volume / avg_volume.replace(0, np.nan)
    cond &= (volume_ratio >= getattr(cfg, "DAYTRADE_VOLUME_RATIO_MIN", 1.2))

    day_range = (high - low).replace(0, np.nan)
    close_position = (close - low) / day_range
    cond &= (close_position >= getattr(cfg, "DAYTRADE_CLOSE_POSITION_MIN", 0.6))

    cond &= (rsi >= getattr(cfg, "DAYTRADE_RSI_MIN", 50.0)) & \
            (rsi <= getattr(cfg, "DAYTRADE_RSI_MAX", 75.0))

    if getattr(cfg, "DAYTRADE_REQUIRE_MACD_HIST_RISING", True):
        cond &= (macd_hist_0 > macd_hist_1)

    return cond.fillna(False)


def compute_daytrade_score_at(ind, date_idx):
    """
    デイトレード候補同士を比較するための勢いスコアを算出する。
    値が大きいほど「本日中に上昇する勢いが強い」とみなす目安（機械的な heuristic）。
    apply_daytrade_filter_at() を通過した候補が埋め合わせ枠より多い場合に、
    スコアの高い順に優先して選ぶために使う。

    重みは固定値の目安であり、運用しながら調整して構わない。
    """
    close = ind["close"].loc[date_idx]
    open_ = ind["open"].loc[date_idx]
    high = ind["high"].loc[date_idx]
    low = ind["low"].loc[date_idx]
    volume = ind["volume"].loc[date_idx]
    avg_volume = ind["avg_volume"].loc[date_idx]
    rsi = ind["rsi"].loc[date_idx]
    macd_hist_0 = ind["macd_hist"].loc[date_idx]
    macd_hist_1 = ind["macd_hist"].shift(1).loc[date_idx]

    # 出来高急増度合い（上限5倍でクリップし、極端な値に引っ張られすぎないようにする）
    volume_ratio = (volume / avg_volume.replace(0, np.nan)).clip(upper=5.0)

    # 当日の値幅の中で、どれだけ高値圏で引けたか（1に近いほど買い優勢）
    day_range = (high - low).replace(0, np.nan)
    close_position = (close - low) / day_range

    # 当日の陽線の実体の大きさ（始値に対する上昇率）
    gain_pct = (close - open_) / open_.replace(0, np.nan)

    # RSIはDAYTRADE_RSI_MIN〜MAXの中央付近を最も高く評価する山型スコアにする
    rsi_min = getattr(cfg, "DAYTRADE_RSI_MIN", 50.0)
    rsi_max = getattr(cfg, "DAYTRADE_RSI_MAX", 75.0)
    rsi_center = (rsi_min + rsi_max) / 2
    rsi_half_width = max((rsi_max - rsi_min) / 2, 1e-9)
    rsi_score = 1 - ((rsi - rsi_center).abs() / rsi_half_width).clip(lower=0, upper=1)

    # MACDヒストグラムの加速度（前日からどれだけ上向きに変化したか）
    macd_accel = (macd_hist_0 - macd_hist_1).clip(lower=-1, upper=1)

    score = (
        volume_ratio.fillna(0) * 1.0
        + close_position.fillna(0) * 2.0
        + gain_pct.fillna(0) * 10.0
        + rsi_score.fillna(0) * 1.0
        + macd_accel.fillna(0) * 1.0
    )
    return score
