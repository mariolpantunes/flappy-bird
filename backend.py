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
        self.HEIGHT = 42
        self.WIDTH = 60
        self.px = px
        self.py = py
        self.v = 0
        self.a = 10
        self.click = False
        self.highscore = 0
        self.done = False
    
    def update(self, dt=1/30):
        if self.click:
            self.click = False
            self.a = max(-100, self.a-100)
            self.v = 0.0
        else:
            self.a = min(10, self.a + 10)
        
        self.v = self.v + self.a * dt
        self.py = self.py + self.v * dt


class World:
    def __init__(self):
        self.HEIGHT = 400
        self.WIDTH = 400
        self.players = {}
        self.pipes = []
        self.highscore = 0
    
    def add_player(self, ws):
        if ws not in self.players:
            self.players[ws] = Player()
    
    def player_click(self, ws):
        if ws in self.players:
            self.players[ws].click = True
    
    def update(self, dt=1/30):
        [p.update(dt) for p in self.players.values()]
        [p.update() for p in self.pipes]
    
    def collisions(self):
        # collisions with world

        keys_to_remove = []

        for k in self.players:
            p = self.players[k]
            if p.py < 0 or (p.py+p.HEIGHT) > self.HEIGHT:
                logger.info('collision...')
                #p.py = 0 if p.py < 0 else self.HEIGHT-p.HEIGHT
                #p.dead(self.highscore)
                keys_to_remove.append(k)
        return keys_to_remove

    def update_highscore(self):
        self.highscore += 1
    
    def dump(self):
        return {'evt':'world_state',
        'highscore':self.highscore,
        'players':[{'px':p.px,'py':p.py, 'v':p.v, 'a':p.a} for p in self.players.values()]}


class GameServer:
    def __init__(self):
        self.viewers = set()
        #self.players = {}
        self.world = World()

    async def incomming_handler(self, websocket, path):
        try:
            async for message in websocket:
                logger.info(message)
                data = json.loads(message)
                if data['cmd'] == 'join':
                    if path == '/viewer':
                        self.viewers.add(websocket)
                    else:
                        self.world.add_player(websocket)
                
                if data['cmd'] == 'click' and path == '/player':
                    self.world.player_click(websocket)

        except websockets.exceptions.ConnectionClosed as c:
            logger.info('Client disconnected')
            if websocket in self.viewers:
                self.viewers.remove(websocket)
    
    async def mainloop(self):
        while True:
            # Update world state
            self.world.update()
            keys_to_remove = self.world.collisions()
            for k in keys_to_remove:
                await k.send(json.dumps({'evt':'done','highscore':self.world.highscore}))
                self.world.players.pop(k)
            self.world.update_highscore()

            # Get world state
            world_state = json.dumps(self.world.dump())

            #share world state wih all players and viewers:
            viewers_to_remove = []
            for v in self.viewers:
                try:
                    await v.send(world_state)
                except websockets.exceptions.ConnectionClosed as c:
                    viewers_to_remove.append(v)
            self.viewers.difference_update(viewers_to_remove)

            players_to_remove = []
            for p in self.world.players.keys():
                try:
                    await p.send(world_state)
                except websockets.exceptions.ConnectionClosed as c:
                    players_to_remove.append(p)
            [self.world.players.pop(key) for key in players_to_remove]
                        
            await asyncio.sleep(1/30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backend server')
    parser.add_argument('-p', type=int, default=8765, help='server port')
    args = parser.parse_args()

    game = GameServer()
    game_loop_task = asyncio.ensure_future(game.mainloop())
    websocket_server = websockets.serve(game.incomming_handler, 'localhost', 8765)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(websocket_server, game_loop_task))
    loop.close()
