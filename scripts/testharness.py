from googleapiclient import discovery
import json

PROMPT_PATH = './realtoxicityprompts-data/prompts.jsonl'

class TestHarness:
    def __init__(self):
        self.API_KEY = "AIzaSyBhoNi1Om9gxeVIGcMdlLfTCeXYQgb8EU8"
        self.challenging_prompts = []
        self.load_prompts()

    def get_perspective_score(self, text):
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
        response = client.comments().analyze(body=analyze_request).execute()
        return response['attributeScores']['TOXICITY']['summaryScore']['value']

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
