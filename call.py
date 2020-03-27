import flask
from tensorflow.keras import layers , activations , models , preprocessing , utils
from LSTM import str_to_tokens, input_word_dict,max_output_length,max_input_length,output_word_dict
from flask import Flask, request
import tensorflow as tf
import numpy as np
from flask import Flask, request
import tensorflow as tf
import numpy as np
import flask
from algoliasearch.search_client import SearchClient
import firebase_admin
import google.cloud
from firebase_admin import credentials, firestore
current_step = 0
NStep = "step number "
curentRecipe = "current recipe"
app = Flask(__name__)
enc_model = tf.keras.models.load_model('enc_model.h5',)
dec_model = tf.keras.models.load_model('dec_model.h5')
model = tf.keras.models.load_model('model.h5')
import os
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import re
from threading import Thread
from playsound import playsound
import sys
import pyaudio
from word2number import w2n
from six.moves import queue
from google.cloud import texttospeech
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r".\Pakalo-d533aea70f49.json"

import firebase_admin
import google.cloud
from firebase_admin import credentials, firestore
from algoliasearch.search_client import SearchClient

    
class Recipe():
    def __init__(self, recipe_name, recipe_ingredients, recipe_instructions):
        self.recipe_name = recipe_name
        self.recipe_ingredients = recipe_ingredients
        self.recipe_instructions = recipe_instructions
    
    def get_recipe_name(self):
        return self.recipe_name
    
    def get_all_ingredients(self):
        return self.recipe_ingredients
    
    def get_all_instructions(self):
        return self.recipe_instructions
        
    def get_instruction(self, index):
        if index >= len(self.recipe_instructions):
            return "There are only " + str(len(self.recipe_instructions)) + " instructions for this recipe"
        return self.recipe_instructions[index]
    
    def get_ingredient(self, ingredient_name):
        return [ingredient for ingredient in self.recipe_ingredients if ingredient_name in ingredient]



class Recipe_db():
    def __init__(self):
        client = SearchClient.create('RONI3GVMZF', 'd069d6b6b79085dc8fce9e619feed841')
        self.algolia_recipe_db = client.init_index('recipes')
        
        cred = credentials.Certificate("./pakalo-abid786-firebase-adminsdk-xc8ts-ad52022ec7.json")
        app = firebase_admin.initialize_app(cred)
        self.firestore_recipe_db = firestore.client().collection('recipes')
    
    def search(self, recipe_substring):
        '''returns list of tuple of search string with recipe name and objectID'''
        search_results = self.algolia_recipe_db.search(recipe_substring)
        return [(i['recipeTitle'], i["objectID"]) for i in search_results["hits"]]
        
    def get_recipe(self, object_id):
        recipe_dict = self.firestore_recipe_db.document(object_id).get().to_dict()
        return Recipe(recipe_dict['title'], recipe_dict['ingredients'], recipe_dict['instructions'])

recipeDb = Recipe_db()
recipe_selected = None
flag = False
# request model prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        a = request.args['a']
        recipe = {}
        global flag
        if(flag):
            recipe = {"instructions": recipe_selected.get_all_instructions(), "ingredients": recipe_selected.get_all_ingredients(), "title":recipe_selected.get_recipe_name()}
            print(recipe["instructions"])
        if (curentRecipe in a.lower()):
            return str(recipe["title"])
        if(NStep in a.lower()):
            arr = a.lower().split(NStep)
            if(len(arr)>0):
                return str(arr[-1]  )+"th step, " + recipe["instructions"][w2n.word_to_num(arr[-1])]  
        states_values = enc_model.predict( str_to_tokens(a ) )
        empty_target_seq = np.zeros( ( 1 , 1 ) )        
        

        empty_target_seq[0, 0] = output_word_dict['start']
        stop_condition = False
        decoded_translation = ''
            
        while not stop_condition :
          dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
          sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
          sampled_word = None
          for word , index in output_word_dict.items() :
              if sampled_word_index == index :
                  decoded_translation += ' {}'.format( word )
                  sampled_word = word
          
          if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
              stop_condition = True
              
          empty_target_seq = np.zeros( ( 1 , 1 ) )  
          empty_target_seq[ 0 , 0 ] = sampled_word_index
          states_values = [ h , c ] 
        data = decoded_translation.replace(' end', '')
        global current_step
        
        if(data == " first step"):
            current_step = 0
            return "FIRSTLY, "+recipe["instructions"][current_step]

        if(data == " last step"):
            current_step = -1
            return "LASTLY, "+recipe["instructions"][current_step]

        if(data == " next step"):
            current_step+=1
            if(current_step==0):
                return "Recipe Completed"
            return "NOW, "+recipe["instructions"][current_step]

        if(data == " current"):
            return "NOW, "+recipe["instructions"][current_step]
        if(data == " previous step"):
            current_step-=1
            if(current_step)<0:
                return "Recipe just Starts Now. Now previous steps are found"
            return "NOW, "+recipe["instructions"][current_step]

        if(data == " complete recipe" or data == " complete"):
            current_step = 0
            return str(recipe["instructions"])

        if(data == " ingre"):
            current_step = 0
            return str(recipe["ingredients"])
        print(data)  
        return data



@app.route('/getRecipe', methods=['GET', 'POST'])
def getRecipe():
    if request.method == 'GET':
        a = request.args['name']
        recipeNames = ""
        # objects = indexFireStore.search(a)
        
        searches = recipeDb.search(a)
        if(len(searches) == 1):
            global flag
            flag = True
            whole_recipe = recipeDb.get_recipe(searches[0][1])
            global recipe_selected
            recipe_selected = whole_recipe
            whole_recipe_string = whole_recipe.get_recipe_name() + ":"
            
            ingredients = whole_recipe.get_all_ingredients()
            for index, ingredient in enumerate(ingredients):
                whole_recipe_string += ingredient
                if (index < len(ingredients) - 1):
                    whole_recipe_string += ","

            whole_recipe_string += ":"


            instructions = whole_recipe.get_all_instructions()
            for index, instruction in enumerate(instructions):
                whole_recipe_string += instruction
                if (index < len(instructions) - 1):
                    whole_recipe_string += ","
            print(whole_recipe_string)
            return whole_recipe_string
        
        # objectID = objects['hits'][0]['objectID']
        
        # doc_ref = db.collection('recipes').document(objectID)
        # try:
        #     doc = doc_ref.get()
        #     print(u'Document data: {}'.format(doc.to_dict()))
        # except google.cloud.exceptions.NotFound:
        #     print(u'No such document!')
        # recipe = doc.to_dict()
        # print(recipe)

        return_recipes = ""
        for index, recipe in enumerate(searches):
            return_recipes+=recipe[0]
            if (index < len(searches) - 1):
                return_recipes+=","
            
        return return_recipes


@app.route('/textToSpeech', methods=['GET', 'POST'])
def textToSpeech():
    a = request.args['tx']
    threada = Thread(target = customTTS,args = (a,))
    threada.start()
    # customTTS(a)
    return "OK"


def customTTS(a):
   
    print("this Function FGet asssssssssssssssssssssss")
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(text=a)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.types.VoiceSelectionParams(
        language_code='en-IN',
        ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

    # Select the type of audio file you want returned
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(synthesis_input, voice, audio_config)


    # The response's audio_content is binary.
    with open('output.mp3', 'wb') as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')
    playsound('output.mp3')
    os.remove('output.mp3')
    return 
app.run(port = 8000, debug = False, threaded = False)


        

# import flask
# from tensorflow.keras import layers , activations , models , preprocessing , utils
# from LSTM import str_to_tokens, input_word_dict,max_output_length,max_input_length,output_word_dict
# from flask import Flask, request
# import tensorflow as tf
# import numpy as np
# from flask import Flask, request
# import tensorflow as tf
# import numpy as np
# import flask


# app = Flask(__name__)
# enc_model = tf.keras.models.load_model('enc_model.h5',custom_objects=None,compile=True)
# dec_model = tf.keras.models.load_model('dec_model.h5',custom_objects=None,compile=True)
# model = tf.keras.models.load_model('model.h5',custom_objects=None,compile=True)


# graph = tf.get_default_graph()

# # request model prediction
# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'GET':
#         a = request.args['a']
#     global graph
#     with graph.as_default():
#         states_values = enc_model.predict( str_to_tokens(a ) )
#         empty_target_seq = np.zeros( ( 1 , 1 ) )
#         empty_target_seq[0, 0] = output_word_dict['start']
#         stop_condition = False
#         decoded_translation = ''
#         while not stop_condition :
#           dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
#           sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
#           sampled_word = None
#           for word , index in output_word_dict.items() :
#               if sampled_word_index == index :
#                   decoded_translation += ' {}'.format( word )
#                   sampled_word = word
          
#           if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
#               stop_condition = True
              
#           empty_target_seq = np.zeros( ( 1 , 1 ) )  
#           empty_target_seq[ 0 , 0 ] = sampled_word_index
#           states_values = [ h , c ] 
#         data = {'result': decoded_translation.replace(' end', '')}
#         return flask.jsonify(data)
# app.run(debug = False, threaded = False)
        