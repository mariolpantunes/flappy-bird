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
    def __init__(self, px=158, py=140):
        self.px = px
        self.py = py
        self.v = 0
        self.a = 10
        self.click = False
    
    def update(self):
        if self.click:
            self.click = False
            self.a = max(-100, self.a-100)
            #self.v = 0.0
        else:
            self.a = min(10, self.a + 10)
        
        self.v = self.v + self.a * 1/30
        self.py = self.py + self.v * 1/30


class GameServer:
    def __init__(self):
        self.viewers = set()
        self.players = {}

    async def incomming_handler(self, websocket, path):
        try:
            async for message in websocket:
                logger.info(message)
                data = json.loads(message)
                if data['cmd'] == 'join':
                    if path == '/viewer':
                        self.viewers.add(websocket)
                    else:
                        self.players[websocket] = Player()
                
                if data['cmd'] == 'click' and path == '/player':
                    if self.players[websocket]:
                        self.players[websocket].click = True

        except websockets.exceptions.ConnectionClosed as c:
            logger.info('Client disconnected')
            if websocket in self.viewers:
                self.viewers.remove(websocket)
    
    async def mainloop(self):
        while True:
            #logger.info("Waiting for a viewer")
            #self.current_player = await self.players.get()

            # Update world state
            for p in self.players.values():
                p.update()

            # Get world state
            world_state = json.dumps({'evt':'world_state', 'players':[{'px':p.px,'py':p.py, 'v':p.v, 'a':p.a} for p in self.players.values()]})

            #share world state wih all players and viewers:
            viewers_to_remove = []
            for v in self.viewers:
                try:
                    await v.send(world_state)
                except websockets.exceptions.ConnectionClosed as c:
                    viewers_to_remove.append(v)
            self.viewers.difference_update(viewers_to_remove)

            players_to_remove = []
            for p in self.players.keys():
                try:
                    await p.send(world_state)
                except websockets.exceptions.ConnectionClosed as c:
                    players_to_remove.append(p)
            [self.players.pop(key) for key in players_to_remove]
                        
            await asyncio.sleep(1/30)


if __name__ == '__main__':
    game = GameServer()
    game_loop_task = asyncio.ensure_future(game.mainloop())
    websocket_server = websockets.serve(game.incomming_handler, 'localhost', 8765)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(websocket_server, game_loop_task))
    loop.close()
