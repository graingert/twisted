#!/usr/bin/env python3
"""
Reproduce, with pure asyncio (no Twisted), the half-open hang seen in
AbortConnectionTests on the AsyncioSelectorReactor: a flow-controlled loopback
connection is abortively closed (SO_LINGER {1,0} + close) by the client, but
the peer -- watched via loop.add_reader -- never sees the disconnect.

Each iteration mirrors what the asyncio reactor does during
test_fullWriteBuffer, using only the low-level event-loop fd APIs the reactor
itself uses (add_reader/add_writer/remove_reader/remove_writer + raw sockets):

  1. Loopback client + accepted server socket, both non-blocking.
  2. Server "stops reading" (never registered as a reader) so its receive
     window goes to zero as the client fills the buffers.
  3. Client registers as reader+writer and fills until send() blocks.
  4. Client aborts: remove_reader + remove_writer, then (deferred by call_soon,
     as the reactor defers connectionLost) SO_LINGER {1,0} + close().
  5. Server "starts reading": registered via add_reader; its callback drains
     the buffered data and then waits for EOF/errno (the disconnect).
  6. If the server callback never reports the disconnect within TIMEOUT, the
     abort was not delivered -- the hang.

DefaultSelector is KqueueSelector on macOS, exactly as the reactor uses.
Expected on Linux: delivered every time. Run on macOS to compare.
"""
import asyncio
import errno
import platform
import selectors
import socket
import struct
import sys
import time

RUN_SECONDS = 180.0
TIMEOUT = 1.0
LINGER_ZERO = struct.pack("ii", 1, 0)


def make_pair():
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(1)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(lsock.getsockname())
    server, _ = lsock.accept()
    lsock.close()
    client.setblocking(False)
    server.setblocking(False)
    return client, server


async def one_iteration(loop):
    client, server = make_pair()
    cfd, sfd = client.fileno(), server.fileno()

    # Client is a reader (as connections are) and fills as a writer until the
    # send buffer blocks against the non-reading server (window -> zero).
    loop.add_reader(cfd, lambda: None)
    fill_done = loop.create_future()
    chunk = b"x" * 65536

    def do_write():
        try:
            while True:
                client.send(chunk)
        except OSError:
            loop.remove_writer(cfd)
            if not fill_done.done():
                fill_done.set_result(None)

    loop.add_writer(cfd, do_write)
    await fill_done

    # Abort: unregister the client, then (deferred, as the reactor defers
    # connectionLost) abortively close and start the server reading.
    loop.remove_reader(cfd)
    loop.remove_writer(cfd)

    got = loop.create_future()

    def on_server_readable():
        try:
            data = server.recv(1 << 20)
        except OSError as e:
            if not got.done():
                got.set_result(errno.errorcode.get(e.args[0], str(e.args[0])))
            return
        if data == b"":
            if not got.done():
                got.set_result("EOF")
        # else: drained buffered data; keep waiting for the disconnect

    def deferred_abort():
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, LINGER_ZERO)
        client.close()
        loop.add_reader(sfd, on_server_readable)

    loop.call_soon(deferred_abort)

    try:
        await asyncio.wait_for(got, TIMEOUT)
        outcome = "delivered"
    except asyncio.TimeoutError:
        outcome = "UNDELIVERED"
    finally:
        try:
            loop.remove_reader(sfd)
        except Exception:
            pass
        server.close()
        try:
            client.close()
        except Exception:
            pass
    return outcome


async def main_coro():
    loop = asyncio.get_running_loop()
    counts = {}
    deadline = time.monotonic() + RUN_SECONDS
    while time.monotonic() < deadline:
        res = await one_iteration(loop)
        counts[res] = counts.get(res, 0) + 1
    return counts


def main():
    print(f"platform: {platform.platform()}  python: {sys.version.split()[0]}")
    print(f"selector: {type(selectors.DefaultSelector()).__name__}")
    print(f"run={RUN_SECONDS}s timeout={TIMEOUT}s\n")

    counts = asyncio.run(main_coro())
    total = sum(counts.values())
    for key, n in sorted(counts.items()):
        print(f"  {n:6d}/{total}  {key}")
    undelivered = counts.get("UNDELIVERED", 0)
    print(f"\n=> total connections = {total}")
    print(f"=> UNDELIVERED (hang) = {undelivered}/{total}")
    sys.exit(1 if undelivered else 0)


if __name__ == "__main__":
    main()
