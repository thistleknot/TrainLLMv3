import subprocess
import time
import openai
import os
import sqlite3
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
from gptcache import cache
#from gptcache.adapter import openai
import json
import requests

