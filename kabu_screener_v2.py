"""
kabu_screener_v2.py
------------------------------------------------------------
全上場銘柄スクリーニング（シンプル版）

■ 抽出条件（すべて満たす銘柄のみ抽出）
  1) 単元価格上限   : 1単元(UNIT_SHARES株)あたりの株価が MAX_UNIT_PRICE 以下
  2) 出来高下限     : VOLUME_MA_WINDOW日平均出来高が MIN_AVG_VOLUME 以上
  3) 前々日の終値が中期線(移動平均線)の上にある
  4) 前日が陰線 かつ 終値が中期線を割り込んでいる
  5) 当日が陽線 かつ 終値が中期線を上回って戻している
  6) 当日の始値が、前日(陰線)の終値より上にある（ギャップアップで寄り付き）

■ 追加の絞り込み（任意・ON/OFF切り替え可能。kabu_config.py参照）
  - RSIが指定レンジ内であること
  - 前日(陰線)の安値がボリンジャーバンド下限に達していたこと
  - 当日のMACDヒストグラムが前日より上昇していること（勢いの再加速）

上記以外のロジック（スコアリング・期待値推定・バックテスト較正・
約定しやすさ・近傍補完等）は今回の条件と無関係なため実装していない。

■ 出力項目（メール本文）
候補として絞り込めた銘柄のみ、以下を表示する。
  - コード / 銘柄名
  - 現在値・取得時刻（候補銘柄だけ直近の分足を取得し直した実勢に近い値）
  - 1単元価格・出来高平均
  - 評価（RSI・MACD・ボリンジャーバンドから機械的に組み立てた
    短い所見。数値そのものではなく「どんな地合いか」を文章にしたもの）
※「評価」はあくまでテクニカル指標の機械的な解釈であり、
  将来の値動きや利益を保証するものではありません。

■ 重要な注意
本スクリーニングはあくまで過去データに基づく統計的な目安であり、
将来の値動きや利益を保証するものではありません。
------------------------------------------------------------
"""

import warnings

import pandas as pd

import kabu_common as kc
import kabu_config as cfg
import kabu_notify

warnings.simplefilter("ignore")


def screen_stocks(tickers, name_map=None):
    """
    全銘柄の株価データを取得し、指定条件をすべて満たす銘柄を抽出して
    結果のリスト(dict)を返す。
    name_map: {コード(str): 銘柄名(str)} 。渡さない場合、銘柄名は空欄になる。
    """
    name_map = name_map or {}
    symbols = [f"{t}.T" for t in tickers]
    data = kc.download_bulk_data(symbols, period=cfg.HIST_PERIOD)
    if data.empty:
        print("株価データの取得に失敗しました。")
        return []

    closes, opens = data["Close"], data["Open"]
    highs, lows, volumes = data["High"], data["Low"], data["Volume"]

    if len(closes) < cfg.MIN_BARS_REQUIRED:
        print("取得できた日足データが不足しているため、スクリーニングできません。")
        return []

    ind = kc.compute_all_indicators(closes, opens, highs, lows, volumes)
    latest = closes.index[-1]

    cond_hard = kc.apply_hard_filter_at(ind, latest)
    cond_pattern = kc.apply_pattern_filter_at(ind, latest)
    cond_indicators = kc.apply_indicator_filters_at(ind, latest)

    final_cond = (cond_hard & cond_pattern & cond_indicators).fillna(False)
    matched_symbols = list(ind["close"].loc[latest][final_cond].dropna().index)

    if not matched_symbols:
        return []

    # 候補として絞り込めた銘柄（数が少ない）だけ、直近の分足で
    # 「現在値」を取得し直す。日足の当日終値には数分〜十数分の
    # 遅延がありうるため、これでより実勢に近い値にする。
    latest_quotes = kc.fetch_latest_quotes(matched_symbols)

    results = [
        _build_candidate_row(symbol, ind, latest, name_map, latest_quotes)
        for symbol in matched_symbols
    ]

    # 出来高が多い順（約定しやすい順の簡易的な目安）に並べる
    results.sort(key=lambda r: r["出来高平均"] or 0, reverse=True)
    return results


def _build_candidate_row(symbol, ind, latest, name_map, latest_quotes):
    """1銘柄分の情報をまとめた、メール表示用の結果行(dict)を組み立てる。"""

    def _at(key, shift=0, default=None):
        series = ind[key]
        if shift:
            series = series.shift(shift)
        v = series.loc[latest, symbol]
        return float(v) if pd.notna(v) else default

    code = symbol.replace(".T", "")
    daily_close = _at("close")

    quote = latest_quotes.get(symbol)
    if quote is not None:
        price = quote["price"]
        price_time = quote["time"].strftime("%Y-%m-%d %H:%M")
    else:
        # 分足の再取得に失敗した場合は、日足の当日終値にフォールバックする
        price = daily_close
        price_time = f"{latest.strftime('%Y-%m-%d')}（日足終値・再取得失敗）"

    macd_hist = ind["macd_hist"].loc[latest, symbol]
    macd_hist_prev = ind["macd_hist"].shift(1).loc[latest, symbol]
    rsi_val = ind["rsi"].loc[latest, symbol]
    bb_lower = ind["bb_lower"].loc[latest, symbol]
    bb_upper = ind["bb_upper"].loc[latest, symbol]

    evaluation = kc.build_evaluation_text(
        rsi=float(rsi_val) if pd.notna(rsi_val) else None,
        macd_hist=float(macd_hist) if pd.notna(macd_hist) else None,
        macd_hist_prev=float(macd_hist_prev) if pd.notna(macd_hist_prev) else None,
        close=daily_close,
        bb_lower=float(bb_lower) if pd.notna(bb_lower) else None,
        bb_upper=float(bb_upper) if pd.notna(bb_upper) else None,
    )

    return {
        "コード": code,
        "銘柄名": name_map.get(code, ""),
        "現在値": round(price, 1) if price is not None else None,
        "取得時刻": price_time,
        "1単元価格": round(price * cfg.UNIT_SHARES, 0) if price is not None else None,
        "出来高平均": _safe_int(ind["avg_volume"].loc[latest, symbol]),
        "評価": evaluation,
    }


def _safe_round(val, digits):
    return round(float(val), digits) if pd.notna(val) else None


def _safe_int(val):
    return int(val) if pd.notna(val) else None


def _print_active_filters():
    print("=== 有効な条件 ===")
    print(f"  単元価格上限        : {cfg.MAX_UNIT_PRICE:,}円以下")
    print(f"  出来高下限          : 直近{cfg.VOLUME_MA_WINDOW}日平均 {cfg.MIN_AVG_VOLUME:,}株以上")
    print(f"  中期線              : {cfg.MID_TERM_MA_WINDOW}日移動平均線")
    print(f"  パターン            : 前々日終値>中期線 → 前日陰線で中期線割れ → "
          f"当日陽線で中期線を回復（当日始値は前日終値より上）")
    print(f"  RSIフィルタ         : {'ON' if cfg.ENABLE_RSI_FILTER else 'OFF'}"
          + (f" ({cfg.RSI_MIN}〜{cfg.RSI_MAX})" if cfg.ENABLE_RSI_FILTER else ""))
    print(f"  ボリンジャーバンド  : {'ON' if cfg.ENABLE_BB_FILTER else 'OFF'}"
          + (" (前日安値がBB下限以下)" if cfg.ENABLE_BB_FILTER and cfg.BB_REQUIRE_PREV_LOW_TOUCH_LOWER else ""))
    print(f"  MACDフィルタ        : {'ON' if cfg.ENABLE_MACD_FILTER else 'OFF'}"
          + (" (ヒストグラムが前日より上昇)" if cfg.ENABLE_MACD_FILTER and cfg.MACD_REQUIRE_HIST_RISING else ""))
    print()


if __name__ == "__main__":
    if not kc.is_market_open_now() and 0:
        print("現在は取引時間外のため、スクリーニングをスキップします。")
    else:
        _print_active_filters()

        ticker_master = kc.get_japanese_ticker_master()

        if not ticker_master.empty:
            all_tickers = ticker_master["コード"].tolist()
            name_map = dict(zip(ticker_master["コード"], ticker_master["銘柄名"]))

            matched_stocks = screen_stocks(all_tickers, name_map=name_map)

            print("\n=== スクリーニング結果 ===")
            if not matched_stocks:
                print("条件に合致する銘柄は見つかりませんでした。")
            else:
                print(f"該当銘柄数: {len(matched_stocks)}件\n")
                for stock in matched_stocks[:cfg.MAX_WRITE_COUNT]:
                    print(stock)
                if len(matched_stocks) > cfg.MAX_WRITE_COUNT:
                    print(f"...他 {len(matched_stocks) - cfg.MAX_WRITE_COUNT} 件（表示省略）")

            if getattr(cfg, "ENABLE_EMAIL_NOTIFY", False):
                if matched_stocks or getattr(cfg, "ENABLE_EMAIL_ON_EMPTY", True):
                    kabu_notify.send_screening_result_email(matched_stocks)
