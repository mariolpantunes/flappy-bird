#!/usr/bin/env python


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


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
    '''
    Data class used to hold the rectangle data.
    Used in the collision algorithm.
    '''
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


def rect_rect_collision(r1:Rect, r2:Rect) -> bool:
    '''
    Computes the collision of two rectangular shapes.
    
    Args:
        r1 (Rect): the first rectangle
        r2 (Rect): the second rectangle

    Returns:
        bool: true in case of collision, false otherwise
    '''
    return r1.x < r2.x + r2.w and r1.x + r1.w > r2.x and r1.y < r2.y + r2.h and r1.h + r1.y > r2.y


class Player:
    '''
    Class that represents a player in the game (a flappy bird).
    '''
    def __init__(self, uuid:str, px:int=200, py:int=140) -> None:
        '''
        Constructor for a Player object.

        Args:
            uuid (str): unique identifier
            px (int): initial position x (default 200)
            py (int): initial position y (default 140)
        '''
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
    
    def update(self, dt:float=1/30) -> None:
        '''
        Updates the player position.

        Args:
            dt (float): the duration of each frame (default 1/30)
        '''
        if self.click:
            self.click = False
            self.v = -25
            self.a = 0
        else:
            self.a = 10
        
        self.v = self.v + self.a * dt
        self.py = self.py + self.v * dt


class Pipe:
    '''
    Class that represents a pipe in the game.
    '''
    def __init__(self, px:int, py:int) -> None:
        '''
        Constructor for a Pipe object.

        Args:
            px (int): initial position x
            py (int): initial position y
        '''
        self.HEIGHT = 355 
        self.WIDTH = 60
        self.GAP = 100
        self.HEAD = 30
        self.px = px
        self.py_top = py
        self.py_bottom = py+self.GAP
        self.v = -10
    
    def update(self, dt:float=1/30) -> None:
        '''
        Updates the pipe position.
        
        Args:
            dt (float): the duration of each frame (default 1/30)
        '''
        self.px = self.px + self.v * dt
    
    def collisions(self, players:list, world_height:int) -> list:
        '''
        Check if any player collided with the pipe.

        Args:
            players (list): list of active players
            world_height (int): the world height
        
        Returns:
            list: players that collided with the pipe
        '''
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
    '''
    Class that contains all the elements in the world (birds and pipes).
    '''
    def __init__(self, with_pipes:bool=False) -> None:
        '''
        Constructor for a World object.

        Args:
            with_pipes (bool): with or without pipes
        '''
        self.HEIGHT = 400
        self.WIDTH = 580
        self.players = {}
        self.pipes = []
        self.with_pipes = with_pipes
        self.highscore = 0
        self.generation = 0
        # OPTIONAL
        self.neural_network = None
    
    def reset(self) -> None:
        '''
        Reset the world state.
        '''
        self.highscore = 0
        self.generation += 1
        self.pipes.clear()

    def add_player(self, ws, uuid:str) -> None:
        '''
        Adds a new player to the world.

        Args:
            ws: websocket that identifies the players
            uuid (str): player unique identification
        '''
        if ws not in self.players:
            self.players[ws] = Player(uuid)
    
    def number_players(self) -> int:
        '''
        Returns the current number of active players.

        Returns:
            int: number of active players
        '''
        return len(self.players)
    
    def player_click(self, ws) -> None:
        '''
        Flag the player input (click).

        Args:
            ws: websocket that identifies the players
        '''
        if ws in self.players:
            self.players[ws].click = True
            
    
    def player_neural_network(self, neural_network:dict) -> None:
        '''
        Neural network definition and activation to be displayed on the viewer.

        Args:
            neural_network (dict): neural network definition and activation data
        '''
        if len(self.players) == 1:
            self.neural_network = neural_network
    
    def update(self, dt:float=1/30) -> None:
        '''
        Updates the position of the pipes and the players.

        Args:
            dt (float): the duration of each frame (default 1/30)
        '''
        dt *= 6.0
        [p.update(dt) for p in self.players.values()]
        
        if self.with_pipes:
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

    
    def collisions(self) -> list:
        '''
        Check if the players have colidded with the pipes or the world

        Returns:
            list: player to be removed
        '''
        keys_to_remove = set()
        
        if self.with_pipes:
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

    def update_highscore(self, dt:float=1/30) -> None:
        '''
        Update the highscore

        Args:
            dt (float): the duration of each frame (default 1/30)
        '''
        self.highscore += dt
    
    def dump(self) -> dict:
        '''
        Dumps the world content into a dictionary.

        Returns:
            dict: world content 
        '''
        players = {}
        for p in self.players.values():
            players[p.uuid] = {'px':p.px,'py':p.py,'v':p.v,'a':p.a}
        rv = {'evt':'world_state',
        'highscore':self.highscore,
        'generation':self.generation,
        'players':players,
        'pipes':[{'px':p.px,'py_t':p.py_top,'py_b':p.py_bottom} for p in self. pipes]}
        if self.neural_network:
            rv['neural_network'] = self.neural_network
        return rv


class GameServer:
    '''
    Class that manages the game and the incoming messages.
    '''
    def __init__(self, with_pipes=False):
        '''
        Constructor for a GameServer object.

        Args:
            with_pipes (bool): with or without pipes
        '''
        self.viewers = set()
        self.world = World(with_pipes=with_pipes)

    async def incomming_handler(self, websocket, path:str) -> None:
        '''
        Manages the incomming messages.

        Args:
            websocket: websocket
            path (str): path used by the client
        '''
        try:
            async for message in websocket:
                data = json.loads(message)
                if data['cmd'] == 'join':
                    if path == '/viewer':
                        self.viewers.add(websocket)
                    else:
                        self.world.add_player(websocket, data['id'])
                
                if data['cmd'] == 'click' and path == '/player':
                    self.world.player_click(websocket)
                
                if data['cmd'] == 'neural_network' and path == '/player':
                    self.world.player_neural_network(data['neural_network'])
                
                if data['cmd'] == 'training' and path == '/training':
                    # send the data to the viewers
                    data['evt'] = 'training'
                    viewers_to_remove = []
                    for v in self.viewers:
                        try:
                            await v.send(json.dumps(data))
                        except websockets.exceptions.ConnectionClosed as c:
                            viewers_to_remove.append(v)
                    self.viewers.difference_update(viewers_to_remove)

        except websockets.exceptions.ConnectionClosed as c:
            logger.info('Client disconnected')
            if websocket in self.viewers:
                self.viewers.remove(websocket)
            # TODO: remove players
            #elif websocket in self.
    
    async def mainloop(self, args: argparse.Namespace) -> None:
        '''
        Defines the main loop of the game.

        Args:
            args (argparse.Namespace): the arguments to configure the game loop
        '''
        while True:
            logger.info(f'Reset game')
            self.world.reset()
            logger.info(f'Waiting for {args.n} players')
            done = False
            while not done:
                # check if the have all the players
                if self.world.number_players() >= args.n:
                    done = True
                else:
                    await asyncio.sleep(1)
            logger.info(f'Starting the main loop')
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
                elif args.l > 0 and  self.world.highscore >= args.l:
                    done = True
                    # remove all the players
                    players_to_remove = []
                    players_to_remove.extend(self.world.players.keys())
                    for p in players_to_remove:
                        await p.send(json.dumps({'evt':'done','highscore':self.world.highscore}))
                        self.world.players.pop(p)
                delay = max(0, 1/args.f-(time.perf_counter() - t))
                await asyncio.sleep(delay)


async def main(args: argparse.Namespace) -> None:
    '''
    Main (async) method.

    Setups the game loop and the websocketserver.

    Args:
        args (argparse.Namespace): the program arguments
    '''
    random.seed(args.s)
    with_pipes = True if args.pipes is not None else False
    game = GameServer(with_pipes=with_pipes)
    websocket_server = websockets.serve(game.incomming_handler, 'localhost', args.p)
    game_loop_task = asyncio.create_task(game.mainloop(args))
    await asyncio.gather(websocket_server, game_loop_task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backend server')
    parser.add_argument('-p', type=int, default=8765, help='server port')
    parser.add_argument('-s', type=int, default=42, help='random seed')
    parser.add_argument('-f', type=int, default=30, help='server fps')
    parser.add_argument('-n', type=int, default=1, help='concurrent number of players')
    parser.add_argument('-l', type=int, default=-1, help='limit the highscore')
    parser.add_argument('--pipes', action='store_true', help='add pipes to the world')
    args = parser.parse_args()

    asyncio.run(main(args))
