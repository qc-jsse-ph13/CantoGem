import os
from flask import *
import random
from melodygenerator import make_melody_response, Voice, change_tempo, mg_list

from paths import PROGRESS_PATH, MIDI_SAVE_PATH, MP3_PATH, get_user_data_path

app = Flask(__name__)

user_is_generating = set()

for key in mg_list.keys():
    make_melody_response("大江東去,", Voice.TENOR, filename=key)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/<userid>/generate", methods=["PUT"])
def send_melody(userid):
    if userid in user_is_generating:
        abort(409)

    data = request.json
    if len(data["lyrics"]) > 150:
        abort(403)

    if "model_type" not in data:
        data["model_type"] = "model_weights"
    
    if data["model_type"] == "random":
        data["model_type"] = random.choice([key for key in mg_list.keys() if key != "model_weights"])
    
    user_is_generating.add(userid)
        
    result = make_melody_response(data["lyrics"], Voice.TENOR, userid, data["model_type"])
    user_is_generating.remove(userid)
    return result

@app.errorhandler(400)
def invalid_text(_):
    return "invalid text input!", 400

@app.route("/<userid>/get-midi")
def send_midi(userid):
    if not os.path.exists(get_user_data_path(userid, MIDI_SAVE_PATH)):
        abort(404)
    return send_file(
        get_user_data_path(userid, MIDI_SAVE_PATH),
        mimetype="audio/midi",
        as_attachment=True
    )

@app.route("/<userid>/get-mp3")
def send_mp3(userid):
    if not os.path.exists(get_user_data_path(userid, MP3_PATH)):
        abort(404)
    return send_file(
        get_user_data_path(userid, MP3_PATH),
        mimetype="audio/mp3",
        as_attachment=True
    )


@app.route("/<userid>/progress")
def get_progress(userid):
    try:
        with open(get_user_data_path(userid, PROGRESS_PATH), "r") as progressbar_file:
            return progressbar_file.read()
    except:
        return "0.00%"


@app.route("/<userid>/changetempo",  methods=["PUT"])
def metric_modulation(userid):
    try:
        data = request.json
        new_tempo = data["new_tempo"]
        change_tempo(userid, int(new_tempo))
        return "good"
    except:
        abort(409)

@app.route("/get-model-list")
def get_model_list():
    return "".join(key for key in mg_list.keys() if key != "model_weights")