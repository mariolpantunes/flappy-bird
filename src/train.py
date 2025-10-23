#!/usr/bin/env python


__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import math
import uuid
import json
import logging
import argparse
import statistics
import numpy as np
import src.nn as nn
import pyBlindOpt.de as de
import pyBlindOpt.ga as ga
import pyBlindOpt.gwo as gwo
import pyBlindOpt.pso as pso
import pyBlindOpt.egwo as egwo
import pyBlindOpt.init as init


from websockets.sync.client import connect


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
    gwo = 'gwo'
    pso = 'pso'
    egwo = 'egwo'

    def __str__(self):
        return self.value


NN_ARCHITECTURE = [
    {'input_dim': 4, 'output_dim': 5, 'activation': 'sigmoid'}
]


def player_game(model: nn.NN) -> float:
    '''
    Player main loop.

    Args:
        model (nn.NN): the model used for playing
    
    Returns:
        float: the highscore achieved
    '''
    identification = str(uuid.uuid4())[:8]
    
    player_x, player_y = 0,0
    pipe_x, pipe_y_t, pipe_y_b = 0,0,0

    
    with connect(f'{CONSOLE_ARGUMENTS.u}/player') as websocket:
        websocket.send(json.dumps({'cmd':'join', 'id':identification}))
        done = False
        while not done:
            data = json.loads(websocket.recv())
            if data['evt'] == 'world_state':
                player = data['players'][identification]

                # find closest pipe
                closest_pipe = None
                for pipe in data['pipes']:
                    if pipe['px']+60 > player['px']:
                        closest_pipe = pipe
                        break

                # Store data
                player_x = player['px']
                player_y = player['py']
                pipe_x = pipe['px']
                pipe_y_t = pipe['py_t']
                pipe_y_b = pipe['py_b']
                
                c =  pipe['py_t'] + pipe['py_b'] / 2
                X = np.array([player['py'], player['v'], c, pipe['px']])
                p = model.predict(X)
                if p[0] > 0.5:
                    websocket.send(json.dumps({'cmd':'click'}))
            elif data['evt'] == 'done':
                d1 = math.sqrt((player_x-pipe_x)**2 + (player_y-pipe_y_b)**2)
                d2 = math.sqrt((player_x-pipe_x)**2 + (player_y-pipe_y_t)**2)
                alpha = 0.1 
                return data['highscore']-alpha*(d1+d2)


def objective(p: np.ndarray) -> float:
    '''
    Objective function used to evaluate the candidate solution.

    Args:
        p (np.ndarray): the parameters of the candidate solution
    
    Returns:
        float: the cost value
    '''
    model = nn.NN(NN_ARCHITECTURE)
    model.update(p)

    highscore = player_game(model)
    return -highscore


def share_training_data(epoch:int, obj:list) -> None:
    '''
    Method that sends the training data to the viewer. 

    Args:
        epoch (int): the current epoch
        obj (list): list with the current objective values
    '''
    # compute the worst, best and average fitness
    worst = max(obj)
    best = min(obj)
    mean = statistics.mean(obj)

    with connect(f'{CONSOLE_ARGUMENTS.u}/training') as websocket:
        websocket.send(json.dumps({'cmd':'training', 'epoch':epoch, 'worst':worst,'best':best, 'mean':mean}))


def callback(epoch:int, obj:list, pop:list) -> None:
    '''
    Callback used to share the training data to the viewer.

    Args:
        epoch (int): the current epoch
        obj (list): list with the current objective values
    '''
    share_training_data(epoch, obj)


def store_data(model:dict, parameters:np.ndarray, path:str) -> None:
    '''
    Store the model into a json file.

    Args:
        model (dict): the model definition
        parameters (np.ndarray): the model parameters
        path (str): the location of the file
    '''
    with open(path, 'w') as f:
        json.dump({'model':model, 'parameters':parameters.tolist()}, f)


def main(args: argparse.Namespace) -> None:
    '''
    Main method.

    Args:
        args (argparse.Namespace): the program arguments
    '''
    # Define the random seed
    np.random.seed(args.s)

    # Define the bounds for the optimization
    bounds = np.asarray([[-1.0, 1.0]]*nn.network_size(NN_ARCHITECTURE))
    
    # Generate the initial population
    population = [nn.NN(NN_ARCHITECTURE).ravel() for i in range(args.n)]

    # Apply Opposition Learning to the inital population
    #population = init.opposition_based(objective, bounds, population=population, n_jobs=args.n)
    #population = init.round_init(objective, bounds, n_pop=args.n, n_rounds=5, n_jobs=args.n)
    population = init.oblesa(objective,bounds=bounds, n_pop=args.n)

    # Run the optimization algorithm
    if args.a is Optimization.de:
        best, _ = de.differential_evolution(objective, bounds, variant='best/3/exp', callback = callback,
        population=population, n_iter=args.e, n_jobs=args.n, cached=False, verbose=True, seed=args.s)
    elif args.a is Optimization.ga:
        best, _ = ga.genetic_algorithm(objective, bounds, n_iter=args.e, callback = callback,
        population=population, n_jobs=args.n, cached=False, verbose=True, seed=args.s)
    elif args.a is Optimization.pso:
        best, _ = pso.particle_swarm_optimization(objective, bounds, n_iter=args.e, callback = callback,
        population=population, n_jobs=args.n, cached=False, verbose=True, seed=args.s)
    elif args.a is Optimization.gwo:
        best, _ = gwo.grey_wolf_optimization(objective, bounds, n_iter=args.e, callback = callback,
        population=population, n_jobs=args.n, cached=False, verbose=True, seed=args.s)
    elif args.a is Optimization.egwo:
        best, _ = egwo.grey_wolf_optimization(objective, bounds, n_iter=args.e, callback = callback,
        population=population, n_jobs=args.n, cached=False, verbose=True, seed=args.s)

    # store the best model
    store_data(NN_ARCHITECTURE, best, args.o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the agents', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-u', type=str, help='server url', default='ws://localhost:8765')
    parser.add_argument('-s', type=int, help='Random generator seed', default=42)
    parser.add_argument('-e', type=int, help='optimization epochs', default=30)
    parser.add_argument('-n', type=int, help='population size', default=10)
    parser.add_argument('-a', type=Optimization, help='Optimization algorithm', choices=list(Optimization), default='pso')
    parser.add_argument('-o', type=str, help='store the best model', default='out/model.json')
    args = parser.parse_args()
    # Global variable for the arguments (required due to the callback function)
    CONSOLE_ARGUMENTS = args

    main(args)