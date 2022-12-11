// Initial setup
const bird_img = new Image();
const back_img = new Image();
const pipe_t_img = new Image();
const pipe_b_img = new Image();
let i = 0;

const epochs = [];
const min_layer = [];
const mean_layer = [];
const max_layer = [];

let line_chart = null;

function init() {
    bird_img.src = 'data/bird.png';
    back_img.src = 'data/back.png';
    pipe_t_img.src = 'data/pipe_top.png';
    pipe_b_img.src = 'data/pipe_bottom.png';

    // check if the plot canvas exists
    let canvas_plot = document.getElementById('canvas_plot');
    if(canvas_plot) {
        // Setup the Line chart
        const ctx = canvas_plot.getContext('2d');
        line_chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [{label: 'Worst',data: min_layer},{label: 'Average',data: mean_layer},{label: 'Best',data: max_layer}]
            },
            options: {
                responsive: false,
                maintainAspectRatio: false,
            }
          });
    }
    
}

init();

const socket = new WebSocket('ws://localhost:8765/viewer');

const players = [];
const pipes = [];

let animation_position = {};

const player_v = 60;
const background_v = 25;

let bx = 0;
let scrollSpeed = 5;

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
    
    if(data.evt == 'reset') {
        generation = 0;
        epochs.length = 0;
        min_layer.length = 0;
        mean_layer.length = 0;
        max_layer.length = 0;
    } else if(data.evt == 'training') {
        if (line_chart != null) {
            console.log('plot into line chart...');
            epochs.push(data.epoch);
            min_layer.push(data.worst);
            mean_layer.push(data.mean);
            max_layer.push(data.best);
            console.log('epochs: '+epochs)
            line_chart.update();
        }
    } else if(data.evt == 'world_state') {
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
            } else {
                let previous_i = (animation_position[key] ?? 0);
                new_animation_position[key] = previous_i;
            }
        });
        animation_position = new_animation_position;

        // Draw the scene
        requestAnimationFrame(draw);

        // OPTIONAL -- get the neural network
        if (data.hasOwnProperty('neural_network') && document.getElementById('canvas_network')) {
            requestAnimationFrame(function() {
                draw_network(data['neural_network'].networkLayer, data['neural_network'].activations);
            });
        }
    }
});


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
