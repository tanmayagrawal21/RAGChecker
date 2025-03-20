"""
FILENAME: proto.py
AUTHOR: Tanmay Agrawal
PURPOSE: Convert a LLM prompt context as well as the output into a list of claims, which are modeled as a triple of (subject, relation, object), and compare the claims with the ground truth claims.
"""


# Make a simple prompt to GPT-4

# Importing the required libraries
import openai
import json
import re
import os

from dotenv import load_dotenv
load_dotenv()


# Setting the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

