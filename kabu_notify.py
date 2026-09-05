"""
kabu_notify.py
------------------------------------------------------------
スクリーニング結果をGmail経由でメール送信するモジュール。

■ 事前準備（初回のみ）
1. 送信に使うGoogleアカウントで2段階認証を有効にする
   https://myaccount.google.com/security

2. 「アプリパスワード」を発行する（通常のログインパスワードとは別物）
   https://myaccount.google.com/apppasswords
   → 発行される16桁の文字列（スペース入り）を使う

3. アプリパスワードと送信元アドレスは、kabu_config.py に直接書かず
   環境変数で渡すことを推奨（Gitやスクリプトの共有時に漏洩しないため）。

   例（Linux/Mac, ターミナルで一時的に設定する場合）:
     export GMAIL_ADDRESS="your_address@gmail.com"
     export GMAIL_APP_PASSWORD="xxxxxxxxxxxxxxxx"

   例（cronから実行する場合）:
     crontab -e で環境変数を書くか、
     ラッパースクリプト(.sh)内で export してから python を呼ぶ

   ※どうしても kabu_config.py に直接書きたい場合は、
     GMAIL_ADDRESS / GMAIL_APP_PASSWORD をそこに定義しても動作する
     （このモジュールは 環境変数 → kabu_config.py の順で探しにいく）
------------------------------------------------------------
"""

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import kabu_config as cfg

GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587


def _build_html_body(matched_stocks):
    """スクリーニング結果(dictのリスト)からメール本文(HTML)を組み立てる。"""
    if not matched_stocks:
        return "<p>本日、条件に合致する銘柄はありませんでした。</p>"

    headers = list(matched_stocks[0].keys())
    header_html = "".join(
        f"<th style='padding:4px 8px;border:1px solid #ddd;background:#f2f2f2;'>{h}</th>"
        for h in headers
    )
    rows_html = ""
    for row in matched_stocks:
        cells = "".join(
            f"<td style='padding:4px 8px;border:1px solid #ddd;'>{row[h]}</td>"
            for h in headers
        )
        rows_html += f"<tr>{cells}</tr>"

    return f"""
    <html>
      <body style="font-family:sans-serif;">
        <p>該当銘柄数: {len(matched_stocks)}件</p>
        <table style="border-collapse:collapse;font-size:13px;">
          <tr>{header_html}</tr>
          {rows_html}
        </table>
        <p style="color:#888;font-size:11px;margin-top:12px;">
          ※本メールは過去データに基づく統計的な目安であり、
          将来の値動きや利益を保証するものではありません（投資は自己責任で）。
        </p>
      </body>
    </html>
    """


def _get_setting(env_name, config_name, default=None):
    """環境変数 → kabu_config.py → default の順で設定値を探す。"""
    return os.environ.get(env_name) or getattr(cfg, config_name, default)


def send_screening_result_email(matched_stocks):
    """
    スクリーニング結果をGmail経由でメール送信する。
    設定不足の場合は送信をスキップしてFalseを返す（例外で落とさない）。
    """
    gmail_address = _get_setting("GMAIL_ADDRESS", "GMAIL_ADDRESS")
    app_password = _get_setting("GMAIL_APP_PASSWORD", "GMAIL_APP_PASSWORD")
    to_addrs = getattr(cfg, "EMAIL_TO", None) or ([gmail_address] if gmail_address else [])

    if not gmail_address or not app_password:
        print("[kabu_notify] GMAIL_ADDRESS / GMAIL_APP_PASSWORD が未設定のため、メール送信をスキップします。")
        return False
    if not to_addrs:
        print("[kabu_notify] 送信先(EMAIL_TO)が未設定のため、メール送信をスキップします。")
        return False

    today_str = datetime.now().strftime("%Y-%m-%d")
    subject_prefix = getattr(cfg, "EMAIL_SUBJECT_PREFIX", "[株スクリーニング]")
    subject = f"{subject_prefix} {today_str} 該当{len(matched_stocks)}件"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(_build_html_body(matched_stocks), "html"))

    try:
        with smtplib.SMTP(GMAIL_SMTP_HOST, GMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(gmail_address, app_password)
            server.sendmail(gmail_address, to_addrs, msg.as_string())
        print(f"[kabu_notify] メールを送信しました（宛先: {', '.join(to_addrs)}）")
        return True
    except Exception as e:
        print(f"[kabu_notify] メール送信に失敗しました: {e}")
        return False
