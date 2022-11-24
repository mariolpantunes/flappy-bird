const socket = new WebSocket('ws://localhost:8765/viewer');
const socket_player = new WebSocket('ws://localhost:8765/player');

const players = [];
const pipes = [];

let animation_position = {};

const player_v = 60;
const background_v = 25;

let bx = 0;
let scrollSpeed = 5;

//let click = false;

let fps = 0;
let alive = 0;
let highscore = 0;
let generation = 0;
let lastTime = performance.now();

// Connection opened
socket.addEventListener('open', (event) => {
    socket.send(JSON.stringify({'cmd':'join'}));
});

// Listen for messages
socket.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    //console.log('Message from server ', data);
    if(data.evt == 'world_state') {
        // Update the world state
        world_state_players = data['players'];
        world_state_pipes = data['pipes'];
        // Get the players information
        if(Object.keys(world_state_players).length > 0) {
            players.length = 0;
            for (let k in world_state_players) {
                let player = world_state_players[k];
                players.push([k, player.px, player.py, player.v]);
            }
        }
        // Get the pipes information
        if(Object.keys(world_state_pipes).length > 0) {
            pipes.length = 0;
            for (let k in world_state_pipes) {
                let pipe = world_state_pipes[k];
                pipes.push([pipe.px, pipe.py_t, pipe.py_b]);
            }
        }
        // Get the debug information
        alive = players.length;
        highscore = Math.round(data['highscore']);
        generation = data['generation'];
        // Get the animation position of each bird
        let new_animation_position = {}
        players.forEach(function(p){
            let key = p[0];
            if(p[3]<0) {
                let previous_i = (animation_position[key] ?? 0) + player_v * (1.0/fps);
                new_animation_position[key] = Math.round(previous_i) < 3 ? previous_i : 0;
                //(previous_i + 1) % 3;
            } else {
                let previous_i = (animation_position[key] ?? 0);
                new_animation_position[key] = previous_i;
            }
        });
        console.log(new_animation_position);
        animation_position = new_animation_position;

        // Draw the scene
        requestAnimationFrame(draw);
    }
});


function onMouseClick() {
    console.log('onMouseClick');
    click = true;
}

/*
// Connection opened
socket_player.addEventListener('open', (event) => {
    socket_player.send(JSON.stringify({'cmd':'join'}));
    const canvas = document.getElementById('canvas');
    canvas.addEventListener('mousedown', onMouseClick);
});

// Listen for messages
socket_player.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    //console.log('Message from server ', data);
    
    if(data.evt == 'world_state') {
        // Update the world state
        if (click == true) {
            socket_player.send(JSON.stringify({'cmd':'click'}));
            click = false;
        }
    }
});*/

const bird_img = new Image();
const back_img = new Image();
const pipe_t_img = new Image();
const pipe_b_img = new Image();
let i = 0;

function init() {
    bird_img.src = 'data/bird.png';
    back_img.src = 'data/back.png';
    pipe_t_img.src = 'data/pipe_top.png';
    pipe_b_img.src = 'data/pipe_bottom.png';
}

function draw() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.globalCompositeOperation = 'destination-over';
    ctx.clearRect(0, 0, canvas.width, canvas.height); 

    // write FPS
    ctx.font = '18px Arial';
    ctx.fillText('FPS: '+fps, 0, 18);
    
    // write highscore
    ctx.fillText('Highscore: '+highscore, 0, 36);

    // write the number of birds that are alive
    ctx.fillText('Alive: '+alive, 0, 54);

    // write the number of generation
    ctx.fillText('Generation: '+generation, 0, 72);

    // Draw Tubes
    pipes.forEach(pipe => {
        // top pipe
        ctx.drawImage(pipe_t_img, pipe[0], pipe[1]-pipe_t_img.height);
    
        // bottom pipe
        ctx.drawImage(pipe_b_img, pipe[0], pipe[2]);
    });

    // Draw players
    players.forEach(player => {
        let i = Math.round(animation_position[player[0]])*(bird_img.width/3);
        ctx.drawImage(bird_img, i, 0, bird_img.width/3, bird_img.height, player[1], player[2], bird_img.width/3, bird_img.height);
    });
    
    // TODO: fix this
    //i = (i+(bird_img.width/3))%bird_img.width;
    
    // Draw Infinitely Scrolling Background
    // draw image 1
    bx_int = Math.round(bx);
    ctx.drawImage(back_img, back_img.width-bx_int, 0);
    // draw image 2
    ctx.drawImage(back_img, -bx_int, 0);
    // update image height
    bx = bx + background_v*(1/fps);//scrollSpeed;
    //resetting the images when the first image entirely exits the screen
    if (bx >= back_img.width) {bx = 0;}

    // Update FPS counter
    const currentTime = performance.now();
    fps = Math.round(1000 / (currentTime - lastTime));
    lastTime = currentTime;
}

init();