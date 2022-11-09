#!/usr/bin/env python


import math
import json
import asyncio
import logging
import threading
import websockets


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

wslogger = logging.getLogger('websockets')
wslogger.setLevel(logging.WARN)


class Player:
    def __init__(self, px, py):
        self.px = 0
        self.py = 0


class GameServer:
    def __init__(self):
        self.viewers = set()
        self.players = {Player(0, 0)}

    async def incomming_handler(self, websocket, path):
        try:
            async for message in websocket:
                logger.info(message)
                data = json.loads(message)
                if data['cmd'] == 'join':
                    if path == '/viewer':
                        self.viewers.add(websocket)

        except websockets.exceptions.ConnectionClosed as c:
            logger.info('Client disconnected')
            if websocket in self.viewers:
                self.viewers.remove(websocket)
    
    async def mainloop(self):
        while True:
            logger.info("Waiting for a viewer")
            #self.current_player = await self.players.get()

            if self.players:
                for p in self.players:
                    p.px = (p.px + 10)%400
                    p.py = 200*math.sin(p.px/100)+200
            
                    if self.viewers:
                        for v in self.viewers:
                            await v.send(json.dumps({'player':0,'px':p.px,'py':p.py}))
            await asyncio.sleep(1/10)

if __name__ == "__main__":
    game = GameServer()
    game_loop_task = asyncio.ensure_future(game.mainloop())
    websocket_server = websockets.serve(game.incomming_handler, 'localhost', 8765)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(websocket_server, game_loop_task))
    loop.close()
