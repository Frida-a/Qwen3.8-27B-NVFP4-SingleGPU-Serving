"""Least-outstanding-requests HTTP reverse proxy over N backends (one vLLM server per GPU).

usage: python3 lb.py <listen_port> <backend_port> [<backend_port> ...]
GET /health is 200 only when every backend is healthy, so `until curl -sf .../health` waits for all.
Responses (including SSE streams) are passed through unchanged.
"""
import asyncio
import sys

import aiohttp
from aiohttp import web

PORT, BACKENDS = int(sys.argv[1]), [int(p) for p in sys.argv[2:]]
inflight = {p: 0 for p in BACKENDS}
HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}


async def health(request):
    async with aiohttp.ClientSession() as s:
        for p in BACKENDS:
            try:
                async with s.get(f"http://127.0.0.1:{p}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status != 200:
                        return web.Response(status=503, text=f"backend {p}: {r.status}")
            except Exception as e:
                return web.Response(status=503, text=f"backend {p}: {e!r}")
    return web.json_response({"status": "ok", "inflight": inflight})


async def proxy(request):
    p = min(BACKENDS, key=lambda b: inflight[b])
    inflight[p] += 1
    try:
        body = await request.read()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        sess = request.app["session"]
        async with sess.request(request.method, f"http://127.0.0.1:{p}{request.path_qs}",
                                data=body, headers=headers) as r:
            resp = web.StreamResponse(status=r.status,
                                      headers={k: v for k, v in r.headers.items() if k.lower() not in HOP})
            await resp.prepare(request)
            async for chunk in r.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    finally:
        inflight[p] -= 1


async def make_app():
    app = web.Application(client_max_size=1 << 30)
    app["session"] = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
        connector=aiohttp.TCPConnector(limit=0))
    app.router.add_get("/health", health)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=PORT, access_log=None)
