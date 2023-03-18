from googleapiclient import discovery
from googleapiclient.errors import HttpError
import json
import time

PROMPT_PATH = './realtoxicityprompts-data/prompts.jsonl'

class TestHarness:
    def __init__(self):
        self.API_KEY = "AIzaSyBhoNi1Om9gxeVIGcMdlLfTCeXYQgb8EU8"
        self.challenging_prompts = []
        self.load_prompts()
        self.last_called = time.time()

    def get_perspective_score(self, text):

        time.sleep(max(0, 1.1-(time.time()-self.last_called)))
        client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=self.API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
        )
        analyze_request = {
        'comment': { 'text': f'{text}' },
        'requestedAttributes': {'TOXICITY': {}}
        }
        try:
            self.last_called = time.time()
            response = client.comments().analyze(body=analyze_request).execute()
        except HttpError as error:
            print(f"An error occurred: {error}")
            return -1
        score = response['attributeScores']['TOXICITY']['summaryScore']['value']
        # print(score)
        return score

    def load_prompts(self):
        json_list = []

        with open(PROMPT_PATH, 'r') as json_file:
            json_list = list(json_file)
            
        challenging_prompts = []
        for json_str in json_list:
            res = json.loads(json_str)
            if res['challenging'] == True:
                challenging_prompts.append(res)

        self.challenging_prompts = challenging_prompts[1:]
        #print(challenging_prompts[15:20])
