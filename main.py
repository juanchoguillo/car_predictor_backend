# env set up with SQLAlchemy db
from io import BytesIO
import pickle
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os 
# install marshmallow_sqlalchemy
from flask_restx import Api, Resource, fields
import joblib
# pip install scikit-learn

#for model 
import pandas as pd
from joblib import load
import requests


app = Flask(__name__)

basedir=os.path.abspath(os.path.dirname(__file__))
# print(basedir)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+os.path.join(basedir, 'db.sqlite' )
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app=app)
ma = Marshmallow(app=app)


# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Car Predictor Api',
    description='Car Predictor Api')

ns = api.namespace('predict', 
     description='Car Predictor')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'YEAR', 
    type=str, 
    required=True, 
    help='year of the vehicle', 
    location='args')
parser.add_argument(
    'MILEAGE', 
    type=str, 
    required=True, 
    help='mileage of the vehicle', 
    location='args')
parser.add_argument(
    'STATE', 
    type=str, 
    required=True, 
    help='state of the vehicle', 
    location='args')
parser.add_argument(
    'MAKE', 
    type=str, 
    required=True, 
    help='make of the vehicle', 
    location='args')
parser.add_argument(
    'MODEL', 
    type=str, 
    required=True, 
    help='model of the vehicle', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

# Definición de la clase para disponibilización
@ns.route('/')
class CarPredictionApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_car_price(args['YEAR'], args['MILEAGE'], args['STATE'], args['MAKE'], args['MODEL'])
        }, 200




def predict_car_price(year, mileage, state, make, model):
    model_in_number = get_model_index(model)
    state_in_number = get_state_index(state)
    make_in_number = get_make_index(make)
    data = {'Year': [year], 'Mileage': [mileage], 'State2': [state_in_number], 'Make2': [make_in_number], 'Model2': [model_in_number]}
    input_df = pd.DataFrame(data)
    # model_in = load('proyecto_rf_final.joblib')
    # AWS linux
    model_in = load('model.joblib')

    price = model_in.predict(input_df)
    return price



# def predict_car_price(year, mileage, state, make, model):
#     model_in_number = get_model_index(model)
#     state_in_number = get_state_index(state)
#     make_in_number = get_make_index(make)
#     data = {'Year': [year], 'Mileage': [mileage], 'State2': [state_in_number], 'Make2': [make_in_number], 'Model2': [model_in_number]}
#     input_df = pd.DataFrame(data)
#     model_url = "https://github.com/juanchoguillo/car_predictor_backend/blob/main/main.py#:~:text=main.py-,proyecto_rf_final,-.joblib"
#     response = requests.get(model_url)

#     if response.status_code == 200:
#         # Save the model file locally
#         with open('model.joblib', 'wb') as file:
#             file.write(response.content)

#         # Load the model from the local file
#         try:
#             model_in = joblib.load('model.joblib')
#         except Exception as e:
#             return {"error": "Failed to load the model.", "details": str(e)}

#         # Use the model for predictions
#         price = model_in.predict(input_df)

#         # Clean up the local model file if needed
#         # ...

#         # Return the predicted price
#         return {"price": price.item()}
#     else:
#         return {"error": "Failed to download the model file."}



def get_state_index(state):
    state_dict = {'Alaska': 0, 'Alabama': 1, 'Arkansas': 2, 'Arizona': 3, 'California': 4, 'Colorado': 5, 'Connecticut': 6, 'District of Columbia': 7, 'Delaware': 8, 'Florida': 9, 'Georgia': 10, 'Hawaii': 11, 'Iowa': 12, 'Idaho': 13, 'Illinois': 14, 'Indiana': 15, 'Kansas': 16, 'Kentucky': 17, 'Louisiana': 18, 'Massachusetts': 19, 'Maryland': 20, 'Maine': 21, 'Michigan': 22, 'Minnesota': 23, 'Missouri': 24, 'Mississippi': 25, 'Montana': 26, 'North Carolina': 27, 'North Dakota': 28, 'Nebraska': 29, 'New Hampshire': 30, 'New Jersey': 31, 'New Mexico': 32, 'Nevada': 33, 'New York': 34, 'Ohio': 35, 'Oklahoma': 36, 'Oregon': 37, 'Pennsylvania': 38, 'Rhode Island': 39, 'South Carolina': 40, 'South Dakota': 41, 'Tennessee': 42, 'Texas': 43, 'Utah': 44, 'Virginia': 45, 'Vermont': 46, 'Washington': 47, 'Wisconsin': 48, 'West Virginia': 49, 'Wyoming': 50}
    # state = state.strip()
    return state_dict[state]    

def get_make_index(make):

    make_dict = {
    'acura': 0,
    'audi': 1,
    'bmw': 2,
    'bentley': 3,
    'buick': 4,
    'cadillac': 5,
    'chevrolet': 6,
    'chrysler': 7,
    'dodge': 8,
    'fiat': 9,
    'ford': 10,
    'freightliner': 11,
    'gmc': 12,
    'honda': 13,
    'hyundai': 14,
    'infiniti': 15,
    'jaguar': 16,
    'jeep': 17,
    'kia': 18,
    'land': 19,
    'lexus': 20,
    'lincoln': 21,
    'mini': 22,
    'mazda': 23,
    'mercedes-benz': 24,
    'mercury': 25,
    'mitsubishi': 26,
    'nissan': 27,
    'pontiac': 28,
    'porsche': 29,
    'ram': 30,
    'scion': 31,
    'subaru': 32,
    'suzuki': 33,
    'tesla': 34,
    'toyota': 35,
    'volkswagen': 36,
    'volvo': 37
    }
    make = make.strip().lower()
    return make_dict[make]    

def get_model_index(model):

    model_dict = {
        '1': 0,
         '15002wd': 1,
         '15004wd': 2,
         '1500laramie': 3,
         '1500tradesman': 4,
         '200lx': 5,
         '200limited': 6,
         '200s': 7,
         '200touring': 8,
         '25002wd': 9,
         '25004wd': 10,
         '3': 11,
         '300300c': 12,
         '300300s': 13,
         '3004dr': 14,
         '300base': 15,
         '300limited': 16,
         '300touring': 17,
         '35004wd': 18,
         '350z2dr': 19,
         '4runner2wd': 20,
         '4runner4wd': 21,
         '4runner4dr': 22,
         '4runnerlimited': 23,
         '4runnerrwd': 24,
         '4runnersr5': 25,
         '4runnertrail': 26,
         '5': 27,
         '500pop': 28,
         '6': 29,
         '7': 30,
         '911': 31,
         '9112dr': 32,
         'a34dr': 33,
         'a44dr': 34,
         'a64dr': 35,
         'a8': 36,
         'acadiaawd': 37,
         'acadiafwd': 38,
         'accent4dr': 39,
         'accord': 40,
         'accordex': 41,
         'accordex-l': 42,
         'accordlx': 43,
         'accordlx-s': 44,
         'accordse': 45,
         'altima4dr': 46,
         'armada2wd': 47,
         'armada4wd': 48,
         'avalanche2wd': 49,
         'avalanche4wd': 50,
         'avalon4dr': 51,
         'avalonlimited': 52,
         'avalontouring': 53,
         'avalonxle': 54,
         'azera4dr': 55,
         'boxster2dr': 56,
         'c-class4dr': 57,
         'c-classc': 58,
         'c-classc300': 59,
         'c-classc350': 60,
         'c702dr': 61,
         'cc4dr': 62,
         'cr-v2wd': 63,
         'cr-v4wd': 64,
         'cr-vex': 65,
         'cr-vex-l': 66,
         'cr-vlx': 67,
         'cr-vse': 68,
         'cr-zex': 69,
         'ct': 70,
         'ctct': 71,
         'cts': 72,
         'cts-v': 73,
         'cts4dr': 74,
         'cx-7fwd': 75,
         'cx-9awd': 76,
         'cx-9fwd': 77,
         'cx-9grand': 78,
         'cx-9touring': 79,
         'caliber4dr': 80,
         'camaro2dr': 81,
         'camaroconvertible': 82,
         'camarocoupe': 83,
         'camry': 84,
         'camry4dr': 85,
         'camrybase': 86,
         'camryl': 87,
         'camryle': 88,
         'camryse': 89,
         'camryxle': 90,
        'canyon2wd': 91,
        'canyon4wd': 92,
        'canyoncrew': 93,
        'canyonextended': 94,
        'cayenneawd': 95,
        'cayman2dr': 96,
        'challenger2dr': 97,
        'challengerr/t': 98,
        'charger4dr': 99,
        'chargerse': 100,
        'chargersxt': 101,
        'cherokeelimited': 102,
        'cherokeesport': 103,
        'civic': 104,
        'civicex': 105,
        'civicex-l': 106,
        'civiclx': 107,
        'civicsi': 108,
        'cobalt2dr': 109,
        'cobalt4dr': 110,
        'colorado2wd': 111,
        'colorado4wd': 112,
        'coloradocrew': 113,
        'coloradoextended': 114,
        'compass4wd': 115,
        'compasslatitude': 116,
        'compasslimited': 117,
        'compasssport': 118,
        'continental': 119,
        'cooper': 120,
        'corolla4dr': 121,
        'corollal': 122,
        'corollale': 123,
        'corollas': 124,
        'corvette2dr': 125,
        'corvetteconvertible': 126,
        'corvettecoupe': 127,
        'cruzelt': 128,
        'cruzesedan': 129,
        'dts4dr': 130,
        'dakota2wd': 131,
        'dakota4wd': 132,
        'durango2wd': 133,
        'durango4dr': 134,
        'durangoawd': 135,
        'durangosxt': 136,
        'e-classe': 137,
        'e-classe320': 138,
        'e-classe350': 139,
        'es': 140,
        'eses': 141,
        'eclipse3dr': 142,
        'econoline': 143,
        'edgelimited': 144,
        'edgese': 145,
        'edgesel': 146,
        'edgesport': 147,
        'elantra': 148,
        'elantra4dr': 149,
        'elantralimited': 150,
        'element2wd': 151,
        'element4wd': 152,
        'enclaveconvenience': 153,
        'enclaveleather': 154,
        'enclavepremium': 155,
        'eos2dr': 156,
        'equinoxawd': 157,
        'equinoxfwd': 158,
        'escalade': 159,
        'escalade2wd': 160,
        'escalade4dr': 161,
        'escaladeawd': 162,
        'escape4wd': 163,
        'escape4dr': 164,
        'escapefwd': 165,
        'escapelimited': 166,
        'escapelimited': 167,
        'escapes': 168,
        'escapese': 169,
        'escapexlt': 170,
        'excursion137"': 171,
        'expedition': 172,
        'expedition2wd': 173,
        'expedition4wd': 174,
        'expeditionlimited': 175,
        'expeditionxlt': 176,
        'explorer': 177,
        'explorer4wd': 178,
        'explorer4dr': 179,
        'explorerbase': 180,
        'explorereddie': 181,
        'explorerfwd': 182,
        'explorerlimited': 183,
        'explorerxlt': 184,
        'express': 185,
        'f-1502wd': 186,
        'f-1504wd': 187,
        'f-150fx2': 188,
        'f-150fx4': 189,
        'f-150king': 190,
        'f-150lariat': 191,
        'f-150limited': 192,
        'f-150platinum': 193,
        'f-150stx': 194,
        'f-150supercrew': 195,
        'f-150xl': 196,
        'f-150xlt': 197,
        'f-250king': 198,
        'f-250lariat': 199,
        'f-250xl': 200,
        'f-250xlt': 201,
        'f-350king': 202,
        'f-350lariat': 203,
        'f-350xl': 204,
        'f-350xlt': 205,
        'fj': 206,
        'fx35awd': 207,
        'fiestas': 208,
        'fiestase': 209,
        'fitsport': 210,
        'flexlimited': 211,
        'flexse': 212,
        'flexsel': 213,
        'focus4dr': 214,
        'focus5dr': 215,
        'focuss': 216,
        'focusse': 217,
        'focussel': 218,
        'focusst': 219,
        'focustitanium': 220,
        'forester2.5x': 221,
        'forester4dr': 222,
        'forte': 223,
        'forteex': 224,
        'fortelx': 225,
        'fortesx': 226,
        'frontier': 227,
        'frontier2wd': 228,
        'frontier4wd': 229,
        'fusion4dr': 230,
        'fusionhybrid': 231,
        'fusions': 232,
        'fusionse': 233,
        'fusionsel': 234,
        'g35': 235,
        'g37': 236,
        'g64dr': 237,
        'gli4dr': 238,
        'gs': 239,
        'gsgs': 240,
        'gti2dr': 241,
        'gti4dr': 242,
        'gx': 243,
        'gxgx': 244,
        'galant4dr': 245,
        'genesis': 246,
        'golf': 247,
        'grand': 248,
        'highlander': 249,
        'highlander4wd': 250,
        'highlander4dr': 251,
        'highlanderbase': 252,
        'highlanderfwd': 253,
        'highlanderlimited': 254,
        'highlanderse': 255,
        'is': 256,
        'isis': 257,
        'impala4dr': 258,
        'impalals': 259,
        'impalalt': 260,
        'impreza': 261,
        'impreza2.0i': 262,
        'imprezasport': 263,
        'jetta': 264,
        'journeyawd': 265,
        'journeyfwd': 266,
        'journeysxt': 267,
        'ls': 268,
        'lsls': 269,
        'lx': 270,
        'lxlx': 271,
        'lacrosse4dr': 272,
        'lacrosseawd': 273,
        'lacrossefwd': 274,
        'lancer4dr': 275,
        'land': 276,
        'legacy': 277,
        'legacy2.5i': 278,
        'legacy3.6r': 279,
        'liberty4wd': 280,
        'libertylimited': 281,
        'libertysport': 282,
        'lucerne4dr': 283,
        'm-classml350': 284,
        'mdx4wd': 285,
        'mdxawd': 286,
        'mkxawd': 287,
        'mkxfwd': 288,
        'mkz4dr': 289,
        'mx5': 290,
        'malibu': 291,
        'malibu1lt': 292,
        'malibu4dr': 293,
        'malibuls': 294,
        'malibult': 295,
        'matrix5dr': 296,
        'maxima4dr': 297,
        'mazda34dr': 298,
        'mazda35dr': 299,
        'mazda64dr': 300,
        'milan4dr': 301,
        'model': 302,
        'monte': 303,
        'murano2wd': 304,
        'muranoawd': 305,
        'muranos': 306,
        'mustang2dr': 307,
        'mustangbase': 308,
        'mustangdeluxe': 309,
        'mustanggt': 310,
        'mustangpremium': 311,
        'mustangshelby': 312,
        'navigator': 313,
        'navigator2wd': 314,
        'navigator4wd': 315,
        'navigator4dr': 316,
        'new': 317,
        'odysseyex': 318,
        'odysseyex-l': 319,
        'odysseylx': 320,
        'odysseytouring': 321,
        'optima4dr': 322,
        'optimaex': 323,
        'optimalx': 324,
        'optimasx': 325,
        'outback2.5i': 326,
        'outback3.6r': 327,
        'outlander': 328,
        'outlander2wd': 329,
        'outlander4wd': 330,
        'pt': 331,
        'pacificalimited': 332,
        'pacificatouring': 333,
        'passat': 334,
        'passat4dr': 335,
        'pathfinder2wd': 336,
        'pathfinder4wd': 337,
        'pathfinders': 338,
        'pathfinderse': 339,
        'patriot4wd': 340,
        'patriotlatitude': 341,
        'patriotlimited': 342,
        'patriotsport': 343,
        'pilot2wd': 344,
        'pilot4wd': 345,
        'pilotex': 346,
        'pilotex-l': 347,
        'pilotlx': 348,
        'pilotse': 349,
        'pilottouring': 350,
        'prius': 351,
        'prius5dr': 352,
        'priusbase': 353,
        'priusfive': 354,
        'priusfour': 355,
        'priusone': 356,
        'priusthree': 357,
        'priustwo': 358,
        'q5quattro': 359,
        'q7quattro': 360,
        'qx562wd': 361,
        'qx564wd': 362,
        'quest4dr': 363,
        'rav4': 364,
        'rav44wd': 365,
        'rav44dr': 366,
        'rav4base': 367,
        'rav4fwd': 368,
        'rav4le': 369,
        'rav4limited': 370,
        'rav4sport': 371,
        'rav4xle': 372,
        'rdxawd': 373,
        'rdxfwd': 374,
        'rx': 375,
        'rx-84dr': 376,
        'rxrx': 377,
        'ram': 378,
        'ranger2wd': 379,
        'ranger4wd': 380,
        'rangersupercab': 381,
        'regal4dr': 382,
        'regalgs': 383,
        'regalpremium': 384,
        'regalturbo': 385,
        'ridgelinertl': 386,
        'ridgelinesport': 387,
        'riolx': 388,
        'roguefwd': 389,
        'rover': 390,
        's2000manual': 391,
        's44dr': 392,
        's60t5': 393,
        's804dr': 394,
        'sc': 395,
        'sl-classsl500': 396,
        'slk-classslk350': 397,
        'srxluxury': 398,
        'sts4dr': 399,
        'santa': 400,
        'savana': 401,
        'sedona4dr': 402,
        'sedonaex': 403,
        'sedonalx': 404,
        'sentra4dr': 405,
        'sequoia4wd': 406,
        'sequoia4dr': 407,
        'sequoialimited': 408,
        'sequoiaplatinum': 409,
        'sequoiasr5': 410,
        'sienna5dr': 411,
        'siennale': 412,
        'siennalimited': 413,
        'siennase': 414,
        'siennaxle': 415,
        'sierra': 416,
        'silverado': 417,
        'sonata4dr': 418,
        'sonatalimited': 419,
        'sonatase': 420,
        'sonichatch': 421,
        'sonicsedan': 422,
        'sorento2wd': 423,
        'sorentoex': 424,
        'sorentolx': 425,
        'sorentosx': 426,
        'soul+': 427,
        'soulbase': 428,
        'sportage2wd': 429,
        'sportageawd': 430,
        'sportageex': 431,
        'sportagelx': 432,
        'sportagesx': 433,
        'sprinter': 434,
        'suburban2wd': 435,
        'suburban4wd': 436,
        'suburban4dr': 437,
        'super': 438,
        'tl4dr': 439,
        'tlautomatic': 440,
        'tsxautomatic': 441,
        'tt2dr': 442,
        'tacoma2wd': 443,
        'tacoma4wd': 444,
        'tacomabase': 445,
        'tacomaprerunner': 446,
        'tahoe2wd': 447,
        'tahoe4wd': 448,
        'tahoe4dr': 449,
        'tahoels': 450,
        'tahoelt': 451,
        'taurus4dr': 452,
        'tauruslimited': 453,
        'taurusse': 454,
        'taurussel': 455,
        'taurusho': 456,
        'terrainawd': 457,
        'terrainfwd': 458,
        'tiguan2wd': 459,
        'tiguans': 460,
        'tiguanse': 461,
        'tiguanseL': 462,
        'titan': 463,
        'titan2wd': 464,
        'titan4wd': 465,
        'touareg4dr': 466,
        'town': 467,
        'transit': 468,
        'traverseawd': 469,
        'traversefwd': 470,
        'tucsonawd': 471,
        'tucsonfwd': 472,
        'tucsonlimited': 473,
        'tundra': 474,
        'tundra2wd': 475,
        'tundra4wd': 476,
        'tundrabase': 477,
        'tundralimited': 478,
        'tundrasr5': 479,
        'veracruzawd': 480,
        'veracruzfwd': 481,
        'versa4dr': 482,
        'versa5dr': 483,
        'vibe4dr': 484,
        'wrxbase': 485,
        'wrxlimited': 486,
        'wrxpremium': 487,
        'wrxsti': 488,
        'wrangler': 489,
        'wrangler2dr': 490,
        'wrangler4wd': 491,
        'wranglerrubicon': 492,
        'wranglersahara': 493,
        'wranglersport': 494,
        'wranglerx': 495,
        'x1xdrive28i': 496,
        'x3awd': 497,
        'x3xdrive28i': 498,
        'x5awd': 499,
        'x5xdrive35i': 500,
        'xc60awd': 501,
        'xc60fwd': 502,
        'xc60t6': 503,
        'xc704dr': 504,
        'xc90awd': 505,
        'xc90fwd': 506,
        'xc90t6': 507,
        'xf4dr': 508,
        'xj4dr': 509,
        'xk2dr': 510,
        'xterra2wd': 511,
        'xterra4wd': 512,
        'xterra4dr': 513,
        'yaris': 514,
        'yaris4dr': 515,
        'yarisbase': 516,
        'yarisle': 517,
        'yukon': 518,
        'yukon2wd': 519,
        'yukon4wd': 520,
        'yukon4dr': 521,
        'tc2dr': 522,
        'xb5dr': 523,
        'xd5dr':524,

    }
    model = model.strip().lower()
    return model_dict[model]    


if __name__ == '__main__':
    #local port 
    # app.run(debug=True, port=5000)

    # ubuntu port 
    # app.run(debug=True, host='172.31.22.90', port=5000)

    #AWS linux port 
    app.run(debug=True, host='172.31.85.0', port=5000)
