#!/usr/bin/env python


import json
import asyncio
import logging
import argparse
import websockets


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

wslogger = logging.getLogger('websockets')
wslogger.setLevel(logging.WARN)


async def ws_comm(u):
    async with websockets.connect(u) as websocket:
        await websocket.send(json.dumps({'cmd':'join'}))
        

        while True:
            #await world state
            world = await websocket.recv()
            #logger.info(world)
            #await input

            #share input
            #if i%30 == 0:
            #await websocket.send(json.dumps({'cmd':'click'}))
            #i += 1


async def read_key_press():
    while True:
        logger.info('read_key_press...')


async def main(args, loop):
    ws_communication = loop.create_task(ws_comm(args.u))
    read_input = loop.create_task(read_key_press())
    await asyncio.wait([read_input,ws_communication])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manual player agent')
    parser.add_argument('-u', type=str, default='ws://localhost:8765/player', help='server url')
    args = parser.parse_args()
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args, loop))
    loop.close()

    #asyncio.run(main(args))