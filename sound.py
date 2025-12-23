# -*- coding: utf-8 -*-
"""
iFLYTEK TTS (WebSocket v2) 批量生成麻将牌报牌音频：0.wav ~ 41.wav + 吃碰杠胡
修复点：
1) aue="raw" 返回 PCM，用 wave 写 WAV 头
2) 鉴权 URL query urlencode
3) tte="UTF8" 避免无意义音频
4) 增加 ping 保活 + 超时重试 + 轻微延迟，解决 server read msg timeout
依赖：pip install websocket-client
"""

import os
import json
import time
import base64
import hashlib
import hmac
import wave
from datetime import datetime, timezone
from wsgiref.handlers import format_date_time
from urllib.parse import urlencode

import websocket


APPID = "c31e2d82"
APIKey = "fd3256d24b5714060fbcc8220e65d6d0"
APISecret = "M2IzNGE1MDEwZDRkNzQwMGEyOThjN2Jj"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
os.makedirs(OUT_DIR, exist_ok=True)

TILE_VOICE_MAP = {
    "chi.wav": "吃！",
    "peng.wav": "碰！",
    "gang.wav": "杠！",
    "hu.wav": "胡！",

    # "0.wav": "一万", "1.wav": "二万", "2.wav": "三万", "3.wav": "四万",
    # "4.wav": "五万", "5.wav": "六万", "6.wav": "七万", "7.wav": "八万", "8.wav": "九万",
    #
    # "9.wav": "一筒", "10.wav": "二筒", "11.wav": "三筒", "12.wav": "四筒",
    # "13.wav": "五筒", "14.wav": "六筒", "15.wav": "七筒", "16.wav": "八筒", "17.wav": "九筒",
    #
    # "18.wav": "一索", "19.wav": "二索", "20.wav": "三索", "21.wav": "四索",
    # "22.wav": "五索", "23.wav": "六索", "24.wav": "七索", "25.wav": "八索", "26.wav": "九索",
    #
    # "27.wav": "东风", "28.wav": "南风", "29.wav": "西风", "30.wav": "北风",
    # "31.wav": "红中",

    "32.wav": "发财", "33.wav": "白板",

    "34.wav": "梅", "35.wav": "兰", "36.wav": "竹", "37.wav": "菊",
    "38.wav": "春", "39.wav": "夏", "40.wav": "秋", "41.wav": "冬",
}


def make_auth_url() -> str:
    host = "tts-api.xfyun.cn"
    path = "/v2/tts"
    base_url = f"wss://{host}{path}"

    now_utc = datetime.now(timezone.utc)
    date = format_date_time(time.mktime(now_utc.timetuple()))

    signature_origin = f"host: {host}\n" \
                       f"date: {date}\n" \
                       f"GET {path} HTTP/1.1"

    signature_sha = hmac.new(
        APISecret.encode("utf-8"),
        signature_origin.encode("utf-8"),
        digestmod=hashlib.sha256
    ).digest()

    signature = base64.b64encode(signature_sha).decode("utf-8")

    authorization_origin = (
        f'api_key="{APIKey}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

    query = urlencode({"host": host, "date": date, "authorization": authorization})
    return f"{base_url}?{query}"


def synth_pcm_bytes(
    text: str,
    vcn: str = "x4_xiaoyan",
    rate: int = 16000,
    speed: int = 55,
    pitch: int = 50,
    volume: int = 60,
    retries: int = 3,
) -> bytes:
    """
    合成单条文本，返回 PCM bytes。
    关键增强：ping 保活 + 超时重试，避免偶发 server read msg timeout
    """
    last_err = None

    for attempt in range(1, retries + 1):
        audio_buf = bytearray()
        err_holder = {"err": None}

        def on_open(ws):
            try:
                text_b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
                payload = {
                    "common": {"app_id": APPID},
                    "business": {
                        "aue": "raw",
                        "auf": f"audio/L16;rate={rate}",
                        "vcn": vcn,
                        "speed": speed,
                        "pitch": pitch,
                        "volume": volume,
                        "bgs": 0,
                        "tte": "UTF8",
                    },
                    "data": {"status": 2, "text": text_b64}
                }
                ws.send(json.dumps(payload, ensure_ascii=False))
            except Exception as e:
                err_holder["err"] = RuntimeError(f"on_open 发送失败：{e}")
                ws.close()

        def on_message(ws, message):
            try:
                data = json.loads(message)
            except Exception as e:
                err_holder["err"] = RuntimeError(f"返回 JSON 解析失败：{e}\n原始：{str(message)[:200]}")
                ws.close()
                return

            code = data.get("code", -1)
            if code != 0:
                err_holder["err"] = RuntimeError(f"TTS失败 code={code}, message={data.get('message')}")
                ws.close()
                return

            d = data.get("data") or {}
            if "audio" in d:
                audio_buf.extend(base64.b64decode(d["audio"]))

            if d.get("status") == 2:
                ws.close()

        def on_error(ws, error):
            err_holder["err"] = RuntimeError(f"WebSocket error: {error}")

        ws = websocket.WebSocketApp(
            make_auth_url(),
            on_open=on_open,
            on_message=on_message,
            on_error=on_error
        )

        # ✅ ping_interval/ping_timeout：防止中间链路/VPN/代理导致的“半连接”
        ws.run_forever(ping_interval=10, ping_timeout=5)

        if err_holder["err"] is None and len(audio_buf) > 0:
            return bytes(audio_buf)

        last_err = err_holder["err"] or RuntimeError("未知错误：未收到音频数据")
        # ✅ 轻微退避，下一次重试更稳
        time.sleep(0.6 * attempt)

    raise last_err


def save_pcm_as_wav(pcm_bytes: bytes, wav_path: str, rate: int = 16000):
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)


def main():
    vcn = "x4_xiaoyan"
    rate = 16000

    print(f"输出目录：{OUT_DIR}")

    # # 自检
    # test_path = os.path.join(OUT_DIR, "test.wav")
    # print("生成自检音频 test.wav <- 你好，我是一万。")
    # pcm = synth_pcm_bytes("你好，我是一万。", vcn=vcn, rate=rate)
    # save_pcm_as_wav(pcm, test_path, rate=rate)
    # print(f"自检音频已生成：{test_path}")

    # 生成 0~41
    print("开始批量合成 0.wav ~ 41.wav ...")
    for i in range(32, 42):
        fn = f"{i}.wav"
        text = TILE_VOICE_MAP[fn]
        out_path = os.path.join(OUT_DIR, fn)

        print(f"[{i:02d}/41] 合成 {fn} <- {text}")
        pcm = synth_pcm_bytes(text=text, vcn=vcn, rate=rate)
        save_pcm_as_wav(pcm, out_path, rate=rate)

        # ✅ 给服务端一点喘息 + 避免网络抖动叠加
        time.sleep(0.2)

    # 动作
    print("开始合成动作语音 chi/peng/gang/hu ...")
    for fn in ["chi.wav", "peng.wav", "gang.wav", "hu.wav"]:
        text = TILE_VOICE_MAP[fn]
        out_path = os.path.join(OUT_DIR, fn)

        print(f"合成 {fn} <- {text}")
        pcm = synth_pcm_bytes(text=text, vcn=vcn, rate=rate)
        save_pcm_as_wav(pcm, out_path, rate=rate)
        time.sleep(0.2)

    print("全部生成完成 ✅")
    print(f"文件都在：{OUT_DIR}")


if __name__ == "__main__":
    main()



# # -*- coding: utf-8 -*-
# """
# iFLYTEK TTS (WebSocket v2) 批量生成麻将牌报牌音频：0.wav ~ 41.wav + 吃碰杠胡
# 修复点：
# 1) aue="raw" 返回 PCM，需要用 wave 写 WAV 头
# 2) 鉴权 URL 的 query 必须 urlencode，否则握手 400
# 3) business 中补 tte="UTF8"，否则可能合成“无意义音频”（编码不一致）
# 依赖：pip install websocket-client
# """
#
# import os
# import json
# import time
# import base64
# import hashlib
# import hmac
# import wave
# from datetime import datetime, timezone
# from wsgiref.handlers import format_date_time
# from urllib.parse import urlencode
#
# import websocket
#
#
# # =========================
# # 1) 填你的鉴权信息
# # =========================
# APPID = "c31e2d82"
# APIKey = "fd3256d24b5714060fbcc8220e65d6d0"
# APISecret = "M2IzNGE1MDEwZDRkNzQwMGEyOThjN2Jj"
#
#
# # =========================
# # 2) 输出目录
# # =========================
# OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
# os.makedirs(OUT_DIR, exist_ok=True)
#
#
# # =========================
# # 3) 映射：文件名 -> 文本
# # =========================
# TILE_VOICE_MAP = {
#     # 动作
#     "chi.wav": "吃！",
#     "peng.wav": "碰！",
#     "gang.wav": "杠！",
#     "hu.wav": "胡！",
#
#     # 万 0-8
#     "0.wav": "一万", "1.wav": "二万", "2.wav": "三万", "3.wav": "四万",
#     "4.wav": "五万", "5.wav": "六万", "6.wav": "七万", "7.wav": "八万", "8.wav": "九万",
#
#     # 筒 9-17
#     "9.wav": "一筒", "10.wav": "二筒", "11.wav": "三筒", "12.wav": "四筒",
#     "13.wav": "五筒", "14.wav": "六筒", "15.wav": "七筒", "16.wav": "八筒", "17.wav": "九筒",
#
#     # 索 18-26
#     "18.wav": "一索", "19.wav": "二索", "20.wav": "三索", "21.wav": "四索",
#     "22.wav": "五索", "23.wav": "六索", "24.wav": "七索", "25.wav": "八索", "26.wav": "九索",
#
#     # 风 27-30
#     "27.wav": "东风", "28.wav": "南风", "29.wav": "西风", "30.wav": "北风",
#
#     # 中发白 31-33
#     "31.wav": "红中", "32.wav": "发财", "33.wav": "白板",
#
#     # 花 34-37
#     "34.wav": "梅", "35.wav": "兰", "36.wav": "竹", "37.wav": "菊",
#
#     # 季 38-41
#     "38.wav": "春", "39.wav": "夏", "40.wav": "秋", "41.wav": "冬",
# }
#
#
# # =========================
# # 4) 讯飞鉴权 URL（必须 urlencode）
# # =========================
# def make_auth_url() -> str:
#     host = "tts-api.xfyun.cn"
#     path = "/v2/tts"
#     base_url = f"wss://{host}{path}"
#
#     now_utc = datetime.now(timezone.utc)
#     date = format_date_time(time.mktime(now_utc.timetuple()))
#
#     signature_origin = (
#         f"host: {host}\n"
#         f"date: {date}\n"
#         f"GET {path} HTTP/1.1"
#     )
#
#     signature_sha = hmac.new(
#         APISecret.encode("utf-8"),
#         signature_origin.encode("utf-8"),
#         digestmod=hashlib.sha256
#     ).digest()
#
#     signature = base64.b64encode(signature_sha).decode("utf-8")
#
#     authorization_origin = (
#         f'api_key="{APIKey}", algorithm="hmac-sha256", '
#         f'headers="host date request-line", signature="{signature}"'
#     )
#     authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
#
#     query = urlencode({
#         "host": host,
#         "date": date,
#         "authorization": authorization
#     })
#
#     return f"{base_url}?{query}"
#
#
# # =========================
# # 5) 合成：返回 PCM bytes
# # =========================
# def synth_pcm_bytes(
#     text: str,
#     vcn: str = "xiaoyan",
#     rate: int = 16000,
#     speed: int = 55,
#     pitch: int = 50,
#     volume: int = 60
# ) -> bytes:
#     audio_buf = bytearray()
#     err_holder = {"err": None}
#
#     def on_open(ws):
#         text_b64 = base64.b64encode(text.encode("utf-8")).decode("utf-8")
#
#         payload = {
#             "common": {"app_id": APPID},
#             "business": {
#                 "aue": "raw",
#                 "auf": f"audio/L16;rate={rate}",
#                 "vcn": vcn,
#                 "speed": speed,
#                 "pitch": pitch,
#                 "volume": volume,
#                 "bgs": 0,
#                 "tte": "UTF8",   # ✅关键：文本编码声明，避免“无意义音频”
#             },
#             "data": {
#                 "status": 2,
#                 "text": text_b64
#             }
#         }
#         ws.send(json.dumps(payload, ensure_ascii=False))
#
#     def on_message(ws, message):
#         try:
#             data = json.loads(message)
#         except Exception as e:
#             err_holder["err"] = RuntimeError(f"返回 JSON 解析失败：{e}\n原始：{str(message)[:200]}")
#             ws.close()
#             return
#
#         code = data.get("code", -1)
#         if code != 0:
#             err_holder["err"] = RuntimeError(f"TTS失败 code={code}, message={data.get('message')}")
#             ws.close()
#             return
#
#         d = data.get("data") or {}
#         if "audio" in d:
#             audio_buf.extend(base64.b64decode(d["audio"]))
#
#         if d.get("status") == 2:
#             ws.close()
#
#     def on_error(ws, error):
#         err_holder["err"] = RuntimeError(f"WebSocket error: {error}")
#
#     ws = websocket.WebSocketApp(
#         make_auth_url(),
#         on_open=on_open,
#         on_message=on_message,
#         on_error=on_error
#     )
#
#     ws.run_forever()
#
#     if err_holder["err"] is not None:
#         raise err_holder["err"]
#
#     return bytes(audio_buf)
#
#
# # =========================
# # 6) PCM -> WAV（写 WAV 头）
# # =========================
# def save_pcm_as_wav(pcm_bytes: bytes, wav_path: str, rate: int = 16000):
#     with wave.open(wav_path, "wb") as wf:
#         wf.setnchannels(1)
#         wf.setsampwidth(2)  # 16bit
#         wf.setframerate(rate)
#         wf.writeframes(pcm_bytes)
#
#
# # =========================
# # 7) 批量生成
# # =========================
# def main():
#     vcn = "x4_xiaoyan"
#     rate = 16000
#
#     print(f"输出目录：{OUT_DIR}")
#
#     # # 先做一个自检音频，避免你跑一堆才发现还是怪声
#     # test_path = os.path.join(OUT_DIR, "test.wav")
#     # print("生成自检音频 test.wav <- 你好，我是一万。")
#     # pcm = synth_pcm_bytes("你好，我是一万。", vcn=vcn, rate=rate)
#     # save_pcm_as_wav(pcm, test_path, rate=rate)
#     # print(f"自检音频已生成：{test_path}（先双击听一下，正常再继续）")
#
#     # 生成 0~41
#     print("开始批量合成 0.wav ~ 41.wav ...")
#     for i in range(0, 42):
#         fn = f"{i}.wav"
#         text = TILE_VOICE_MAP.get(fn)
#         if not text:
#             raise RuntimeError(f"缺少映射：{fn}")
#
#         out_path = os.path.join(OUT_DIR, fn)
#         print(f"[{i:02d}/41] 合成 {fn} <- {text}")
#
#         pcm = synth_pcm_bytes(text=text, vcn=vcn, rate=rate)
#         save_pcm_as_wav(pcm, out_path, rate=rate)
#
#     # 生成动作语音
#     print("开始合成动作语音 chi/peng/gang/hu ...")
#     for fn in ["chi.wav", "peng.wav", "gang.wav", "hu.wav"]:
#         text = TILE_VOICE_MAP[fn]
#         out_path = os.path.join(OUT_DIR, fn)
#         print(f"合成 {fn} <- {text}")
#
#         pcm = synth_pcm_bytes(text=text, vcn=vcn, rate=rate)
#         save_pcm_as_wav(pcm, out_path, rate=rate)
#
#     print("全部生成完成 ✅")
#     print(f"文件都在：{OUT_DIR}")
#
#
# if __name__ == "__main__":
#     main()
