#!/usr/bin/env python

import time
import math
import json
import random
import asyncio
import logging
import argparse
import websockets
import dataclasses


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


wslogger = logging.getLogger('websockets')
wslogger.setLevel(logging.WARN)


@dataclasses.dataclass
class Rect:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


def rect_rect_collision(r1, r2):
    return r1.x < r2.x + r2.w and r1.x + r1.w > r2.x and r1.y < r2.y + r2.h and r1.h + r1.y > r2.y


class Player:
    def __init__(self, uuid, px=200, py=140):
        self.HEIGHT = 42
        self.WIDTH = 60
        self.px = px
        self.py = py
        self.v = 0
        self.a = 10
        self.click = False
        self.highscore = 0
        self.done = False
        self.uuid = uuid
    
    def update(self, dt):
        if self.click:
            self.click = False
            self.v = -25
            self.a = 0
        else:
            self.a = 10
        
        self.v = self.v + self.a * dt
        self.py = self.py + self.v * dt
        #logger.info(f'({self.py}, {self.v} {self.a})')


class Pipe:
    def __init__(self, px, py):
        self.HEIGHT = 355 
        self.WIDTH = 60
        self.GAP = 100
        self.HEAD = 30
        self.px = px
        self.py_top = py
        self.py_bottom = py+self.GAP
        self.v = -10
    
    def update(self, dt):
        self.px = self.px + self.v * dt
    
    def collisions(self, players, world_height):
        keys_to_remove = []
        
        rect_top = Rect(self.px, 0, self.WIDTH, self.py_top)
        rect_bottom = Rect(self.px, self.py_bottom, self.WIDTH, world_height-self.py_bottom)

        for k in players:
            p = players[k]
            rect_player = Rect(p.px, p.py, p.WIDTH, p.HEIGHT)

            if rect_rect_collision(rect_player, rect_top) or rect_rect_collision(rect_player, rect_bottom):
                keys_to_remove.append(k)
        
        return keys_to_remove
    
    def __str__(self):
        return f'[{self.px} {self.py_top} {self.py_bottom}]'


class World:
    def __init__(self):
        self.HEIGHT = 400
        self.WIDTH = 580
        self.players = {}
        self.pipes = []
        self.highscore = 0
        self.generation = 0
    
    def reset(self):
        self.highscore = 0
        self.generation += 1
        self.pipes.clear()

    def add_player(self, ws, uuid):
        if ws not in self.players:
            self.players[ws] = Player(uuid)
    
    def number_players(self):
        return len(self.players)
    
    def player_click(self, ws):
        if ws in self.players:
            self.players[ws].click = True
    
    def update(self, dt=1/30):
        dt *= 6.0
        [p.update(dt) for p in self.players.values()]
        
        # generate new pipe
        if len(self.pipes) < 3:
            if self.pipes:
                previous_pipe = self.pipes[-1]
                lower_limit = max(previous_pipe.HEAD,previous_pipe.py_top-(previous_pipe.GAP/2))
                upper_limit = min(self.HEIGHT-previous_pipe.HEAD-previous_pipe.GAP,
                previous_pipe.py_top+(previous_pipe.GAP/2))
                py = random.randint(lower_limit, upper_limit)
                self.pipes.append(Pipe(previous_pipe.px+290, py))
            else:
                self.pipes.append(Pipe(self.WIDTH, 150))
        
        # update the pipes position
        [p.update(dt) for p in self.pipes]

        # check if first pipe can be removed
        if self.pipes:
            if self.pipes[0].px+self.pipes[0].WIDTH <=0:
                self.pipes.pop(0)

    
    def collisions(self):
        keys_to_remove = set()
        
        # collisions with pipes
        if self.pipes:
            # all the players have the same px
            random_player = list(self.players.values())[0]
            # select pipe
            closest_pipe = None
            closest_distance = self.WIDTH
            for pipe in self.pipes:
                if pipe.px+pipe.WIDTH > random_player.px:
                    # pipe in front
                    distance = pipe.px-random_player.px
                    if distance < closest_distance:
                        closest_pipe = pipe
                        closest_distance = distance
            keys_to_remove.update(closest_pipe.collisions(self.players, self.HEIGHT))

        # collisions with world
        for k in self.players:
            p = self.players[k]
            if p.py < 0 or (p.py+p.HEIGHT) > self.HEIGHT:
                #p.py = 0 if p.py < 0 else self.HEIGHT-p.HEIGHT
                #p.dead(self.highscore)
                keys_to_remove.add(k)
        
        return keys_to_remove

    def update_highscore(self, dt=1/30):
        self.highscore += dt
    
    def dump(self):
        players = {}
        for p in self.players.values():
            players[p.uuid] = {'px':p.px,'py':p.py,'v':p.v,'a':p.a}
        return {'evt':'world_state',
        'highscore':self.highscore,
        'generation':self.generation,
        'players':players,
        'pipes':[{'px':p.px,'py_t':p.py_top,'py_b':p.py_bottom} for p in self. pipes]}


class GameServer:
    def __init__(self):
        self.viewers = set()
        #self.players = {}
        self.world = World()

    async def incomming_handler(self, websocket, path):
        try:
            async for message in websocket:
                #logger.info(message)
                data = json.loads(message)
                if data['cmd'] == 'join':
                    if path == '/viewer':
                        self.viewers.add(websocket)
                    else:
                        self.world.add_player(websocket, data['id'])
                
                if data['cmd'] == 'click' and path == '/player':
                    self.world.player_click(websocket)

        except websockets.exceptions.ConnectionClosed as c:
            logger.info('Client disconnected')
            if websocket in self.viewers:
                self.viewers.remove(websocket)
    
    async def mainloop(self, args):
        while True:
            self.world.reset()

            done = False
            while not done:
                # check if the have all the players
                if self.world.number_players() >= args.n:
                    done = True
                else:
                    await asyncio.sleep(1)
            
            done = False
            while not done:
                t = time.perf_counter()
                # Update world state
                self.world.update(1/args.f)
                keys_to_remove = self.world.collisions()
                for k in keys_to_remove:
                    await k.send(json.dumps({'evt':'done','highscore':self.world.highscore}))
                    self.world.players.pop(k)
                self.world.update_highscore(1/args.f)

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
                
                # check if the stopping criteria
                if self.world.number_players() == 0:
                    done = True
                elif self.world.highscore >= args.l:
                    done = True
                    # remove all the players
                    players_to_remove = []
                    players_to_remove.extend(self.world.players.keys())
                    for p in players_to_remove:
                        await p.send(json.dumps({'evt':'done','highscore':self.world.highscore}))
                        self.world.players.pop(p)
                delay = max(0, 1/args.f-(time.perf_counter() - t))
                await asyncio.sleep(delay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backend server')
    parser.add_argument('-p', type=int, default=8765, help='server port')
    parser.add_argument('-f', type=int, default=30, help='server fps')
    parser.add_argument('-n', type=int, default=1, help='concurrent number of players')
    parser.add_argument('-l', type=int, default=30, help='limit the highscore')
    args = parser.parse_args()

    game = GameServer()
    game_loop_task = asyncio.ensure_future(game.mainloop(args))
    websocket_server = websockets.serve(game.incomming_handler, 'localhost', args.p)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(websocket_server, game_loop_task))
    loop.close()
