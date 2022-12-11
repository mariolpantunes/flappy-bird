#!/usr/bin/env python


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import uuid
import json
import asyncio
import logging
import argparse
import statistics
import websockets
import numpy as np
import src.nn as nn
import optimization.de as de
import optimization.ga as ga
import optimization.pso as pso


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

wslogger = logging.getLogger('websockets')
wslogger.setLevel(logging.WARN)


@enum.unique
class Optimization(enum.Enum):
    '''
    Enum data type that represents the optimization algorithm
    '''
    de = 'de'
    ga = 'ga'
    pso = 'pso'

    def __str__(self):
        return self.value


NN_ARCHITECTURE = [
    {'input_dim': 4, 'output_dim': 4, 'activation': 'relu'},
    {'input_dim': 4, 'output_dim': 1, 'activation': 'sigmoid'}
]


async def player_game(perceptron):
    identification = str(uuid.uuid4())[:8]
    async with websockets.connect('ws://localhost:8765/player') as websocket:
        await websocket.send(json.dumps({'cmd':'join', 'id':identification}))
        done = False
        while not done:
            data = json.loads(await websocket.recv())
            if data['evt'] == 'world_state':
                player = data['players'][identification]
                # find closest pipe
                closest_pipe = None
                for pipe in data['pipes']:
                    if pipe['px']+60 > player['px']:
                        closest_pipe = pipe
                        break
                
                c =  pipe['py_t'] + pipe['py_b'] / 2

                X = np.array([player['py'], player['v'], c, pipe['px']])
                p = perceptron.predict(X)
                if p[0] >= 0.5:
                    await websocket.send(json.dumps({'cmd':'click'}))
            elif data['evt'] == 'done':
                return data['highscore']


def objective(p):
    model = nn.NN(NN_ARCHITECTURE)
    model.update(p)
    highscore = asyncio.run(player_game(model))
    return -highscore


async def share_training_data(epoch, obj):
    # compute the worst, best and average fitness
    worst = max(obj)
    best = min(obj)
    mean = statistics.mean(obj)
    async with websockets.connect('ws://localhost:8765/training') as websocket:
        await websocket.send(json.dumps({'cmd':'training', 'epoch':epoch, 'worst':worst,'best':best, 'mean':mean}))


def callback(epoch, obj):
    asyncio.run(share_training_data(epoch, obj))


def store_data(model, parameters, path:str):
    with open(path, 'w') as f:
        json.dump({'model':model, 'parameters':parameters.tolist()}, f)


def main(args):
    # Define the bounds for the optimization
    bounds = np.asarray([[-1.0, 1.0]]*nn.network_size(NN_ARCHITECTURE))
    
    # Generate the initial population
    population = [nn.NN(NN_ARCHITECTURE, seed=args.s).ravel() for i in range(args.p)]
    
    # Run the optimization algorithm
    if args.a is Optimization.de:
        best, _ = de.differential_evolution(objective, bounds, variant='best/1/bin', callback = callback,
        population=population, n_iter=args.l, n_jobs=args.p, cached=False, verbose=True, seed=args.s)
    elif args.a is Optimization.ga:
        best, _ = ga.genetic_algorithm(objective, bounds, n_iter=args.l, callback = callback,
        population=population, n_jobs=args.p, cached=False, verbose=True, seed=args.s)
    elif args.a is Optimization.pso:
        best, _ = pso.particle_swarm_optimization(objective, bounds, n_iter=args.l, callback = callback,
        population=population, n_pop=args.p, n_jobs=args.p, cached=False, verbose=True, seed=args.s)

    # store the best model
    store_data(NN_ARCHITECTURE, best, args.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train agents')
    parser.add_argument('-l', type=int, help='number of loops (iterations)', default=30)
    parser.add_argument('-p', type=int, help='population size', default=10)
    parser.add_argument('-a', type=Optimization, help='Optimization algorithm', choices=list(Optimization), default='de')
    parser.add_argument('-s', type=int, help='Random generator seed', default=42)
    parser.add_argument('-o', type=str, help='store the best model', default='out/model.json')
    args = parser.parse_args()

    main(args)