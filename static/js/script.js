const input_box = document.getElementById('input_box');
const submitButton = document.getElementById('submit');
const fixButton = document.getElementById('fix');
const result_image = document.getElementById('ResultImage');
const image_container = document.getElementById('image_container');
const json_code_textArea = document.getElementById('json_code_textArea');
const loading_spinner = document.getElementById('loading-spinner');
var history_messages = [];


// call api
function gen_layout() {
    disableBtn();
    let input_text = input_box.value;
    fetch('/gen_layout', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: input_text })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status == "success") {
            //output json string
            json_code_textArea.value = data.gen_str;
            // put image
            result_image.src = 'data:image/png;base64,' + data.img_base64;
            // append to historical messages
            history_messages.push([data.prompt_str, data.gen_str]);
        }
        enableBtn();
    });
}


// call api
function fix_layout() {
    if (history_messages.length > 0) {
        disableBtn();
        let input_text = input_box.value;
        fetch('/fix_layout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                                query: input_text ,
                                history_messages : history_messages
                            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status == "success") {
                //output json string
                json_code_textArea.value = data.gen_str;
                // put image
                result_image.src = 'data:image/png;base64,' + data.img_base64;
                // append to historical messages
                history_messages.push([data.prompt_str, data.gen_str]);
            }
            enableBtn();
        });
    } else {
        alert("Please generate a layout at first");
    }
    
}



function enableBtn() {
    image_container.style.display = "flex";
    loading_spinner.style.display = "none";
    // submitButton.textContent = "Submit";
    submitButton.classList.remove('stop_button','fade-loop');
    submitButton.disabled = false;
    fixButton.classList.remove('stop_button','fade-loop');
    fixButton.disabled = false;
}

function disableBtn() {
    image_container.style.display = "none";
    loading_spinner.style.display = "flex";
    // submitButton.textContent = "Stop";
    submitButton.classList.add('stop_button','fade-loop');
    submitButton.disabled = true;
    fixButton.classList.add('stop_button','fade-loop');
    fixButton.disabled = true;
}